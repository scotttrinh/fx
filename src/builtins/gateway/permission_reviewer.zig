const std = @import("std");
const gateway_client = @import("../../gateway/client.zig");
const permission_auto_classifier = @import("../../core/permissions/auto_classifier.zig");
const debug_trace = @import("../../core/shared/debug_trace.zig");
const types = @import("../../core/shared/types.zig");
const vercel_protocol = @import("../../gateway/vercel_protocol.zig");
const vercel_model_policy = @import("../../gateway/vercel_model_policy.zig");
const gateway_decision_model = @import("../../gateway/decision_model.zig");
const token_pass = @import("../../core/gateway/token_pass.zig");

const Allocator = std.mem.Allocator;
const ProviderInput = permission_auto_classifier.ProviderInput;
const Reviewer = permission_auto_classifier.Reviewer;
const TransportOutcome = permission_auto_classifier.TransportOutcome;

const single_transport_attempt: usize = 1;
const reviewer_model = "openai/gpt-5.6-luna";

pub const provider = permission_auto_classifier.Provider{ .review_fn = reviewGateway };

/// Every composition site selects this provider, so interactive, ask, and
/// subagent reviews all choose their model and backend here.
fn reviewGateway(
    _: ?*anyopaque,
    alloc: Allocator,
    input: ProviderInput,
    request: permission_auto_classifier.ReviewRequest,
) anyerror!permission_auto_classifier.ParseOutcome {
    const model_id = selectedReviewerModel(input.reviewer_model, request.review_turn.model);
    var backend = Backend.select(input, model_id) orelse {
        debug_trace.logf("permission", "event=auto_review_backend result=unconfigured model={s}", .{model_id});
        return .{ .invalid = .transport_unconfigured };
    };
    debug_trace.logf("permission", "event=auto_review_backend result={s} model={s}", .{ @tagName(backend), model_id });
    return backend.reviewer(input.cancel_flag).review(alloc, request);
}

/// The review_model setting (or FX_REVIEW_MODEL) names the reviewer. Without
/// one, a session on a Token Pass model is reviewed by a Token Pass model, so
/// its reviews stay on Token Pass credit too.
fn selectedReviewerModel(override: []const u8, session_model: []const u8) []const u8 {
    if (override.len > 0) return override;
    return token_pass.modelForSideCall(.review, session_model, reviewer_model);
}

test "the reviewer follows the override, then Token Pass, then the default" {
    try std.testing.expectEqualStrings("openai/gpt-5.6-luna", selectedReviewerModel("", "anthropic/claude-opus-4.8"));
    try std.testing.expectEqualStrings("deepseek/deepseek-v4.1-flash", selectedReviewerModel("", "zai/glm-5.3"));
    try std.testing.expectEqualStrings("openai/gpt-6-sol", selectedReviewerModel("openai/gpt-6-sol", "zai/glm-5.3"));
}

/// What reviews for a gateway session. The reviewer it hands out points back
/// into it, so it stays in place until the review returns.
const Backend = union(enum) {
    /// A language model on the chat endpoint answers with a tool call.
    chat: ChatConfig,
    /// A decision model on the decision endpoint answers one question.
    decision: gateway_decision_model.Config,

    /// Null when the session cannot reach `model_id`, which leaves the review
    /// unavailable and the action held.
    fn select(input: ProviderInput, model_id: []const u8) ?Backend {
        // A host that manages the credential gives fx no key to send.
        const api_key: ?[]const u8 = if (input.credential_source == .host_managed or input.credential.len == 0)
            null
        else
            input.credential;
        if (!gateway_decision_model.isDecisionModelId(model_id)) return .{ .chat = .{
            .api_key = api_key,
            .team = input.tenant,
            .chat_url = input.endpoint,
            .model = model_id,
        } };
        // The decision endpoint takes only a key fx holds, and sits beside a
        // gateway chat URL.
        return .{ .decision = gateway_decision_model.Config.init(.{
            .api_key = api_key orelse return null,
            .team = input.tenant,
            .chat_url = input.endpoint,
            .model_id = model_id,
        }) orelse return null };
    }

    fn reviewer(self: *Backend, cancel_flag: ?*std.atomic.Value(bool)) Reviewer {
        return switch (self.*) {
            .chat => |*chat| .withTransportModel(
                .{ .context = chat, .build_fn = buildChatReview, .send_fn = sendChatReview },
                cancel_flag,
                Reviewer.default_timeout_ms,
                chat.model,
            ),
            .decision => |*config| .withDecisionModel(config.model(), cancel_flag, Reviewer.default_timeout_ms),
        };
    }
};

/// One language-model reviewer on the gateway chat endpoint. Every slice is
/// borrowed for the length of the review.
const ChatConfig = struct {
    /// Null when fx holds no key for the session.
    api_key: ?[]const u8,
    team: ?[]const u8 = null,
    chat_url: []const u8,
    model: []const u8,
    /// Tests answer the request themselves through these.
    stream_ctx: ?*anyopaque = null,
    stream_fn: StreamFn = streamChatReview,
};

const StreamFn = *const fn (
    ?*anyopaque,
    Allocator,
    gateway_client.StreamRequest,
    std.Io.Clock.Timestamp,
    *std.atomic.Value(bool),
) anyerror!gateway_client.StreamResult;

fn streamChatReview(
    _: ?*anyopaque,
    alloc: Allocator,
    request: gateway_client.StreamRequest,
    deadline: std.Io.Clock.Timestamp,
    cancel_flag: *std.atomic.Value(bool),
) !gateway_client.StreamResult {
    return gateway_client.streamGatewayRequiredToolCompletionBounded(alloc, request, deadline, cancel_flag);
}

fn buildChatReview(
    _: *anyopaque,
    alloc: Allocator,
    model: []const u8,
    tools_json: []const u8,
    instructions: []const types.ChatMessage,
    messages: []const types.ChatMessage,
    target_call_id: []const u8,
    deadline: std.Io.Clock.Timestamp,
    cancel_flag: *std.atomic.Value(bool),
) ![]u8 {
    // No later request shares a review's prefix: the review data sits inside
    // the policy instructions. A cache write would only add its premium.
    return vercel_protocol.buildGatewayPendingToolReviewRequestBodyWithMaxOutputTokens(
        alloc,
        tools_json,
        instructions,
        messages,
        target_call_id,
        .{
            .skip_cache_writes = vercel_model_policy.takesPromptCacheOptions(model),
            .single_system_message = vercel_model_policy.singleSystemMessage(model),
        },
        2048,
        deadline,
        cancel_flag,
    );
}

const OwnedStream = struct {
    stream: gateway_client.StreamResult,
};

fn deinitOwnedStream(raw_ctx: *anyopaque, alloc: Allocator) void {
    const owned: *OwnedStream = @ptrCast(@alignCast(raw_ctx));
    owned.stream.deinit(alloc);
    alloc.destroy(owned);
}

fn sendChatReview(
    raw_ctx: *anyopaque,
    alloc: Allocator,
    model: []const u8,
    payload: []const u8,
    deadline: std.Io.Clock.Timestamp,
    cancel_flag: *std.atomic.Value(bool),
) error{OutOfMemory}!TransportOutcome {
    const chat: *const ChatConfig = @ptrCast(@alignCast(raw_ctx));
    debug_trace.logf(
        "permission",
        "event=auto_review_transport_start model={s} retry_count={d}",
        .{ model, single_transport_attempt },
    );
    if (cancel_flag.load(.seq_cst)) return .cancelled;
    if (chat.chat_url.len == 0) {
        debug_trace.logf("permission", "event=auto_review_transport result=permanent_failure reason=missing_gateway_config", .{});
        return .permanent_failure;
    }

    var delivery = gateway_client.DeliveryCertainty.init();
    var stream = chat.stream_fn(chat.stream_ctx, alloc, .{
        .api_key = chat.api_key,
        .team = chat.team,
        .model = model,
        .retry_count = single_transport_attempt,
        .chat_url = chat.chat_url,
        .payload = payload,
        .delivery = &delivery,
    }, deadline, cancel_flag) catch |err| {
        if (err == error.OutOfMemory) return error.OutOfMemory;
        const failure = gateway_client.RequestFailure.fromError(err, cancel_flag);
        debug_trace.logf(
            "permission",
            "event=auto_review_transport result={s} reason=transport_error error={s}",
            .{ @tagName(failure), @errorName(err) },
        );
        return failed(failure);
    };
    var stream_owned = true;
    defer if (stream_owned) stream.deinit(alloc);

    if (cancel_flag.load(.seq_cst)) {
        return .cancelled;
    }
    if (stream.status != .ok) {
        const failure = gateway_client.RequestFailure.fromStatus(stream.status);
        debug_trace.logf(
            "permission",
            "event=auto_review_transport result={s} http_status={d}",
            .{ @tagName(failure), @backingInt(stream.status) },
        );
        return failed(failure);
    }
    if (stream.completion.finish_reason) |reason| switch (reason) {
        .provider_error => {
            debug_trace.logf(
                "permission",
                "event=auto_review_transport result=transient_failure reason=provider_error",
                .{},
            );
            return .transient_failure;
        },
        .content_filter => {
            debug_trace.logf(
                "permission",
                "event=auto_review_transport result=permanent_failure reason=content_filter",
                .{},
            );
            return .permanent_failure;
        },
        .stop, .length, .tool_calls, .other => {},
    };
    debug_trace.logf(
        "permission",
        "event=auto_review_transport result=completion finish_reason={s} tool_calls={d} content_bytes={d}",
        .{
            if (stream.completion.finish_reason) |reason| @tagName(reason) else "absent",
            stream.completion.tool_calls.len,
            if (stream.completion.content) |content| content.len else 0,
        },
    );

    const owned = try alloc.create(OwnedStream);
    owned.* = .{ .stream = stream };
    stream_owned = false;
    return .{ .completion = .{
        .completion = owned.stream.completion,
        .context = @ptrCast(owned),
        .deinit_fn = deinitOwnedStream,
    } };
}

fn failed(failure: gateway_client.RequestFailure) TransportOutcome {
    return switch (failure) {
        .transient_failure => .transient_failure,
        .permanent_failure => .permanent_failure,
        .timed_out => .timed_out,
        .cancelled => .cancelled,
    };
}

const FakeOutcome = enum {
    valid,
    malformed,
    transient_error,
    permanent_error,
    ambiguous_error,
    timeout,
    cancelled,
    transient_http,
    permanent_http,
};

const FakeStream = struct {
    outcomes: []const FakeOutcome,
    calls: usize = 0,
    expected_model: []const u8 = reviewer_model,
    saw_single_attempt_only: bool = true,
    saw_expected_model_only: bool = true,
    saw_required_tool_payload: bool = true,
    deadlines: [2]?std.Io.Clock.Timestamp = .{ null, null },

    fn execute(
        raw_ctx: ?*anyopaque,
        alloc: Allocator,
        request: gateway_client.StreamRequest,
        deadline: std.Io.Clock.Timestamp,
        _: *std.atomic.Value(bool),
    ) anyerror!gateway_client.StreamResult {
        const self: *FakeStream = @ptrCast(@alignCast(raw_ctx.?));
        if (self.calls < self.deadlines.len) self.deadlines[self.calls] = deadline;
        self.saw_single_attempt_only = self.saw_single_attempt_only and request.retry_count == 1;
        self.saw_expected_model_only = self.saw_expected_model_only and std.mem.eql(u8, request.model, self.expected_model);
        self.saw_required_tool_payload = self.saw_required_tool_payload and
            std.mem.find(u8, request.payload, "permission_decision") != null;
        const outcome = self.outcomes[@min(self.calls, self.outcomes.len - 1)];
        self.calls += 1;
        return switch (outcome) {
            .valid => validStream(alloc),
            .malformed => malformedStream(alloc),
            .transient_error => error.ConnectionResetByPeer,
            .permanent_error => error.AccessDenied,
            .ambiguous_error => {
                request.delivery.?.markPossiblySent();
                return error.ConnectionResetByPeer;
            },
            .timeout => error.Timeout,
            .cancelled => error.Cancelled,
            .transient_http => .{ .status = @fromBackingInt(@intCast(429)) },
            .permanent_http => .{ .status = @fromBackingInt(@intCast(400)) },
        };
    }
};

fn validStream(alloc: Allocator) !gateway_client.StreamResult {
    const generation_id = try alloc.dupe(u8, "gen_01ARZ3NDEKTSV4RRFFQ69G5FAV");
    errdefer alloc.free(generation_id);
    const id = try alloc.dupe(u8, "review_1");
    errdefer alloc.free(id);
    const name = try alloc.dupe(u8, permission_auto_classifier.tool_name);
    errdefer alloc.free(name);
    const arguments = try alloc.dupe(
        u8,
        "{\"risk\":\"low\",\"decision\":\"clear\",\"rationale\":\"Narrow authorized action.\"}",
    );
    errdefer alloc.free(arguments);
    const tool_calls = try alloc.alloc(types.ToolCall, 1);
    errdefer alloc.free(tool_calls);
    tool_calls[0] = .{
        .id = id,
        .name = name,
        .arguments_json = arguments,
    };
    return .{
        .status = .ok,
        .completion = .{
            .tool_calls = tool_calls,
            .finish_reason = .tool_calls,
            .generation_id = generation_id,
        },
    };
}

fn malformedStream(alloc: Allocator) !gateway_client.StreamResult {
    return .{
        .status = .ok,
        .completion = .{
            .content = try alloc.dupe(u8, "not a structured decision"),
            .finish_reason = .stop,
        },
    };
}

fn testRequest() permission_auto_classifier.ReviewRequest {
    const pending = types.ChatMessage{
        .role = .assistant,
        .tool_calls = &.{.{
            .id = "call_1",
            .name = "glob_files",
            .arguments_json = "{\"pattern\":\"*\"}",
        }},
    };
    return .{
        .review_turn = .{
            .model = "openai/gpt-5",
            .pending_assistant = pending,
            .target_call_id = "call_1",
            .origin = .root,
            .trusted_root_context = "User asked to inspect the repository.",
        },
        .targets = &.{},
        .action = .{ .tool = .{
            .tool_name = "glob_files",
            .arguments_json = "{\"pattern\":\"*\"}",
        } },
    };
}

fn testChat(fake: *FakeStream, model: []const u8) ChatConfig {
    return .{
        .api_key = "test-key",
        .chat_url = "https://example.test/chat",
        .model = model,
        .stream_ctx = fake,
        .stream_fn = FakeStream.execute,
    };
}

/// Reviews `testRequest` with `model` on the chat backend, answered by `fake`.
fn reviewChat(
    fake: *FakeStream,
    cancel_flag: ?*std.atomic.Value(bool),
    model: []const u8,
) !permission_auto_classifier.ParseOutcome {
    var backend: Backend = .{ .chat = testChat(fake, model) };
    return backend.reviewer(cancel_flag).review(std.testing.allocator, testRequest());
}

test "the backend follows the reviewer model's kind and what the session can reach" {
    const chat_url = "https://ai-gateway.vercel.sh/v4/ai/language-model";
    const Expected = union(enum) { chat: ?[]const u8, decision: []const u8, unavailable };
    const cases = [_]struct { input: ProviderInput, model: []const u8, expected: Expected }{
        .{ .input = .{ .credential = "gw-key", .endpoint = chat_url }, .model = "openai/gpt-5.6-luna", .expected = .{ .chat = "gw-key" } },
        .{ .input = .{ .credential = "ignored", .credential_source = .host_managed, .endpoint = chat_url }, .model = "openai/gpt-5.6-luna", .expected = .{ .chat = null } },
        .{ .input = .{ .credential = "gw-key", .endpoint = chat_url }, .model = "typesafe-ai/jev", .expected = .{ .decision = "https://ai-gateway.vercel.sh/v4/ai/decision-model" } },
        // A decision model needs a key fx holds and a gateway chat URL.
        .{ .input = .{ .credential_source = .host_managed, .endpoint = chat_url }, .model = "typesafe-ai/jev", .expected = .unavailable },
        .{ .input = .{ .endpoint = chat_url }, .model = "typesafe-ai/jev", .expected = .unavailable },
        .{ .input = .{ .credential = "gw-key", .endpoint = "https://example.test/chat" }, .model = "typesafe-ai/jev", .expected = .unavailable },
    };
    for (cases) |case| {
        const backend = Backend.select(case.input, case.model);
        switch (case.expected) {
            .chat => |api_key| {
                try std.testing.expectEqualStrings(case.model, backend.?.chat.model);
                try std.testing.expectEqualDeep(api_key, backend.?.chat.api_key);
            },
            .decision => |url| {
                try std.testing.expectEqualStrings(case.model, backend.?.decision.model_id);
                try std.testing.expectEqualStrings(url, backend.?.decision.url());
            },
            .unavailable => try std.testing.expect(backend == null),
        }
    }
}

test "gateway review asks GPT-5.6 and later to skip cache writes" {
    var cancel = std.atomic.Value(bool).init(false);
    const deadline = std.Io.Clock.Timestamp.fromNow(std.testing.io, .{
        .clock = .awake,
        .raw = .fromMilliseconds(1000),
    });
    const instructions = [_]types.ChatMessage{.{ .role = .system, .content = "Review the pending call." }};
    const messages = [_]types.ChatMessage{
        .{ .role = .user, .content = "review_context_kind: normal\n" },
        .{ .role = .assistant, .tool_calls = &.{.{ .id = "call", .name = "shell", .arguments_json = "{}" }} },
    };
    for ([_]struct { model: []const u8, skip: bool }{
        .{ .model = reviewer_model, .skip = true },
        .{ .model = "openai/gpt-6.1-sol", .skip = true },
        .{ .model = "openai/gpt-5", .skip = false },
        .{ .model = "anthropic/claude-haiku-5.5", .skip = false },
    }) |case| {
        var unused: u8 = 0;
        const body = try buildChatReview(@ptrCast(&unused), std.testing.allocator, case.model, "[]", &instructions, &messages, "call", deadline, &cancel);
        defer std.testing.allocator.free(body);
        try std.testing.expectEqual(case.skip, std.mem.find(u8, body, "\"promptCacheOptions\":{\"mode\":\"explicit\"}") != null);
        // A review never asks the Gateway to cache.
        try std.testing.expect(std.mem.find(u8, body, "\"caching\"") == null);
    }
}

test "gateway review sends Qwen its instructions as one system message" {
    var cancel = std.atomic.Value(bool).init(false);
    const deadline = std.Io.Clock.Timestamp.fromNow(std.testing.io, .{
        .clock = .awake,
        .raw = .fromMilliseconds(1000),
    });
    const instructions = [_]types.ChatMessage{
        .{ .role = .system, .content = "Review the pending call." },
        .{ .role = .system, .content = "Review policy details." },
    };
    const messages = [_]types.ChatMessage{
        .{ .role = .user, .content = "review_context_kind: normal\n" },
        .{ .role = .assistant, .tool_calls = &.{.{ .id = "call", .name = "shell", .arguments_json = "{}" }} },
    };
    for ([_]struct { model: []const u8, systems: usize }{
        .{ .model = "alibaba/qwen3.8-max", .systems = 1 },
        .{ .model = "zai/glm-5.3", .systems = 2 },
    }) |case| {
        var unused: u8 = 0;
        const body = try buildChatReview(@ptrCast(&unused), std.testing.allocator, case.model, "[]", &instructions, &messages, "call", deadline, &cancel);
        defer std.testing.allocator.free(body);
        try std.testing.expectEqual(case.systems, std.mem.count(u8, body, "{\"role\":\"system\""));
    }
}

test "gateway reviewer transport carries the resolved review-model override" {
    var fake = FakeStream{ .outcomes = &.{.valid}, .expected_model = "openai/gpt-5-alt" };
    var outcome = try reviewChat(&fake, null, "openai/gpt-5-alt");
    defer outcome.deinit(std.testing.allocator);

    switch (outcome) {
        .valid => |result| try std.testing.expectEqual(permission_auto_classifier.Decision.clear, result.decision),
        .evidence_incomplete, .invalid => return error.TestExpectedEqual,
    }
    try std.testing.expectEqual(@as(usize, 1), fake.calls);
    try std.testing.expect(fake.saw_expected_model_only);
    try std.testing.expect(fake.saw_required_tool_payload);
}

test "gateway automatic reviewer transport is single-attempt" {
    var fake = FakeStream{ .outcomes = &.{.valid} };
    var outcome = try reviewChat(&fake, null, reviewer_model);
    defer outcome.deinit(std.testing.allocator);

    switch (outcome) {
        .valid => |result| try std.testing.expectEqual(permission_auto_classifier.Decision.clear, result.decision),
        .evidence_incomplete, .invalid => return error.TestExpectedEqual,
    }
    try std.testing.expectEqual(@as(usize, 1), fake.calls);
    try std.testing.expect(fake.saw_single_attempt_only);
    try std.testing.expect(fake.saw_expected_model_only);
    try std.testing.expect(fake.saw_required_tool_payload);
}

test "permission reviewer retries one transport failure with a fresh deadline" {
    var fake = FakeStream{ .outcomes = &.{ .transient_error, .malformed } };
    const outcome = try reviewChat(&fake, null, reviewer_model);

    try std.testing.expectEqual(
        std.meta.Tag(permission_auto_classifier.ParseOutcome).invalid,
        std.meta.activeTag(outcome),
    );
    // One transport retry, then one malformed-completion retry.
    try std.testing.expectEqual(@as(usize, 3), fake.calls);
    try std.testing.expect(fake.saw_single_attempt_only);
    try std.testing.expect(fake.deadlines[0] != null);
    try std.testing.expect(fake.deadlines[1] != null);
    try std.testing.expect(fake.deadlines[1].?.raw.nanoseconds > fake.deadlines[0].?.raw.nanoseconds);
}

test "gateway automatic reviewer distinguishes transient and permanent HTTP failures" {
    var transient_fake = FakeStream{ .outcomes = &.{ .transient_http, .valid } };
    var transient = try reviewChat(&transient_fake, null, reviewer_model);
    defer transient.deinit(std.testing.allocator);
    // Transient failures retry once and can recover.
    try std.testing.expectEqual(
        std.meta.Tag(permission_auto_classifier.ParseOutcome).valid,
        std.meta.activeTag(transient),
    );
    try std.testing.expectEqual(
        permission_auto_classifier.Decision.clear,
        transient.valid.decision,
    );
    try std.testing.expectEqual(@as(usize, 2), transient_fake.calls);

    var permanent_fake = FakeStream{ .outcomes = &.{ .permanent_http, .valid } };
    const permanent = try reviewChat(&permanent_fake, null, reviewer_model);
    try std.testing.expectEqual(
        std.meta.Tag(permission_auto_classifier.ParseOutcome).invalid,
        std.meta.activeTag(permanent),
    );
    try std.testing.expectEqual(
        permission_auto_classifier.InvalidReason.transport_permanent,
        permanent.invalid,
    );
    try std.testing.expectEqual(@as(usize, 1), permanent_fake.calls);
}

test "gateway transport preserves cancellation timeout transient and permanent outcomes" {
    var fake = FakeStream{ .outcomes = &.{
        .cancelled,
        .timeout,
        .transient_error,
        .permanent_error,
    } };
    var config = testChat(&fake, "openai/gpt-5");
    var cancel_flag = std.atomic.Value(bool).init(false);
    const deadline = std.Io.Clock.Timestamp.fromNow(std.testing.io, .{
        .clock = .awake,
        .raw = .fromMilliseconds(100),
    });
    const expected = [_]std.meta.Tag(permission_auto_classifier.TransportOutcome){
        .cancelled,
        .timed_out,
        .transient_failure,
        .permanent_failure,
    };
    for (expected) |expected_tag| {
        const outcome = try sendChatReview(
            @ptrCast(&config),
            std.testing.allocator,
            "openai/gpt-5",
            "{}",
            deadline,
            &cancel_flag,
        );
        try std.testing.expectEqual(expected_tag, std.meta.activeTag(outcome));
    }
    try std.testing.expectEqual(expected.len, fake.calls);
}

test "gateway automatic reviewer distinguishes timeout permanent failure and cancellation" {
    var timeout_fake = FakeStream{ .outcomes = &.{.timeout} };
    const timed_out = try reviewChat(&timeout_fake, null, reviewer_model);
    try std.testing.expectEqual(
        std.meta.Tag(permission_auto_classifier.ParseOutcome).invalid,
        std.meta.activeTag(timed_out),
    );
    try std.testing.expectEqual(
        permission_auto_classifier.InvalidReason.transport_timed_out,
        timed_out.invalid,
    );
    // The timeout is retried once before the review is declared unavailable.
    try std.testing.expectEqual(@as(usize, 2), timeout_fake.calls);

    var permanent_fake = FakeStream{ .outcomes = &.{ .permanent_error, .valid } };
    const permanent = try reviewChat(&permanent_fake, null, reviewer_model);
    try std.testing.expectEqual(
        std.meta.Tag(permission_auto_classifier.ParseOutcome).invalid,
        std.meta.activeTag(permanent),
    );
    try std.testing.expectEqual(
        permission_auto_classifier.InvalidReason.transport_permanent,
        permanent.invalid,
    );
    try std.testing.expectEqual(@as(usize, 1), permanent_fake.calls);

    var cancelled_fake = FakeStream{ .outcomes = &.{.cancelled} };
    try std.testing.expectError(
        error.Cancelled,
        reviewChat(&cancelled_fake, null, reviewer_model),
    );
    try std.testing.expectEqual(@as(usize, 1), cancelled_fake.calls);

    var cancel_flag = std.atomic.Value(bool).init(true);
    var pre_cancelled_fake = FakeStream{ .outcomes = &.{.valid} };
    try std.testing.expectError(
        error.Cancelled,
        reviewChat(&pre_cancelled_fake, &cancel_flag, reviewer_model),
    );
    try std.testing.expectEqual(@as(usize, 0), pre_cancelled_fake.calls);
}
