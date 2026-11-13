//! AI Gateway's decision models behind fx's `decision.Model` contract.
//!
//! One call posts the state and questions to the gateway's `decision-model`
//! endpoint, the sibling of the `language-model` endpoint chat requests use,
//! and reads back one answer per question. The request and response shapes
//! are the AI SDK gateway provider's.

const std = @import("std");
const decision = @import("../core/decision/decision.zig");
const debug_trace = @import("../core/shared/debug_trace.zig");
const io_mod = @import("../core/shared/io.zig");
const secret = @import("../core/auth/secret.zig");
const testing_listener = @import("../core/shared/testing_listener.zig");
const gateway_client = @import("client.zig");

const Allocator = std.mem.Allocator;

const chat_suffix = "/language-model";
const decision_suffix = "/decision-model";
/// A decision response is a few answers and their metadata. A larger body is
/// not one.
const max_response_bytes: usize = 256 * 1024;

/// The models AI Gateway serves through its decision endpoint. The gateway's
/// public catalog marks them with the `decision` type (`evaluation` before the
/// rename), but the coding-agent catalog fx loads lists language models only,
/// so fx keeps the list here.
const model_ids = [_][]const u8{
    "baseten/elephant-1",
    "baseten/joey-1",
    "baseten/mammoth-1",
    "convaiinnovations/laya",
    "google/gemini-decision-1",
    "liquid/d1",
    "liquid/d1-ultra",
    "microsoft/microsoft-decision-1",
    "nace-ai/drex-max-finance",
    "nace-ai/drex-max-v1.1",
    "nace-ai/drex-v2.0",
    "openai/gpt-6-luna-decisions",
    "parallel/parallel-decisions-fast",
    "perplexity/pplx-decider-flash-v1.2",
    "perplexity/pplx-decider-flash-v1.3",
    "perplexity/pplx-decider-v1.2",
    "perplexity/pplx-decider-v1.3",
    "scaledown/scaledown-classify-1",
    "typesafe-ai/jev",
    "typesafe-ai/jev-1.13.0",
    "typesafe-ai/jev-1.14.0",
};

/// Whether `model_id` names a gateway decision model, not a language model.
pub fn isDecisionModelId(model_id: []const u8) bool {
    for (model_ids) |known| {
        if (std.mem.eql(u8, model_id, known)) return true;
    }
    return false;
}

/// Writes the decision endpoint beside the gateway chat URL into `out` and
/// returns it. Null when `chat_url` is not a gateway chat URL or `out` is too
/// small.
pub fn urlForChatUrl(chat_url: []const u8, out: []u8) ?[]const u8 {
    if (!std.mem.endsWith(u8, chat_url, chat_suffix)) return null;
    const base = chat_url[0 .. chat_url.len - chat_suffix.len];
    return std.mem.print(out, "{s}{s}", .{ base, decision_suffix }) catch null;
}

/// One gateway decision model. Every slice is borrowed for as long as the
/// `decision.Model` from `model` is in use.
pub const Config = struct {
    api_key: []const u8,
    team: ?[]const u8 = null,
    /// The gateway's decision endpoint, from `urlForChatUrl`.
    url: []const u8,
    model_id: []const u8,

    pub fn model(self: *Config) decision.Model {
        return .{
            .provider = "gateway",
            .model_id = self.model_id,
            .supported_question_types = .full,
            .context = self,
            .decide_fn = decide,
        };
    }
};

fn decide(
    raw: *anyopaque,
    alloc: Allocator,
    request: decision.Request,
    call: decision.CallOptions,
) Allocator.Error!decision.ModelOutcome {
    const config: *const Config = @ptrCast(@alignCast(raw));
    if (call.cancel_flag.load(.seq_cst)) return .cancelled;
    if (config.api_key.len == 0 or config.url.len == 0) {
        debug_trace.logf("decision", "event=gateway_decision result=permanent_failure reason=missing_gateway_config", .{});
        return .permanent_failure;
    }

    const payload = try requestBodyAlloc(alloc, request);
    defer alloc.free(payload);
    debug_trace.logf("decision", "event=gateway_decision_start model={s} questions={d} payload_bytes={d}", .{ config.model_id, request.questions.len, payload.len });

    var operation = PostOperation{ .alloc = alloc, .config = config, .payload = payload };
    var answer = gateway_client.runBoundedHttpOperation(HttpAnswer, alloc, call.cancel_flag, call.deadline, &operation) catch |err| {
        if (err == error.OutOfMemory) return error.OutOfMemory;
        const outcome: decision.ModelOutcome = if (err == error.Cancelled or call.cancel_flag.load(.seq_cst))
            .cancelled
        else if (err == error.Timeout)
            .timed_out
        else if (gateway_client.isRetryableGatewayError(err))
            .transient_failure
        else
            .permanent_failure;
        debug_trace.logf("decision", "event=gateway_decision result={s} error={s}", .{ @tagName(outcome), @errorName(err) });
        return outcome;
    };
    defer answer.deinit(alloc);
    if (call.cancel_flag.load(.seq_cst)) return .cancelled;

    if (answer.status != .ok) {
        const code: u16 = @backingInt(answer.status);
        const outcome: decision.ModelOutcome = if (code == 408 or code == 425 or code == 429 or code >= 500)
            .transient_failure
        else
            .permanent_failure;
        debug_trace.logf("decision", "event=gateway_decision result={s} http_status={d}", .{ @tagName(outcome), code });
        return outcome;
    }

    const response = parseResponse(alloc, answer.body) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Malformed => {
            debug_trace.logf("decision", "event=gateway_decision result=malformed_response body_bytes={d}", .{answer.body.len});
            return .malformed_response;
        },
    };
    debug_trace.logf("decision", "event=gateway_decision result=response answers={d} input_tokens={d} output_tokens={d}", .{
        response.answers.len,
        response.usage.input_tokens orelse 0,
        response.usage.output_tokens orelse 0,
    });
    return .{ .response = response };
}

const HttpAnswer = struct {
    status: std.http.Status,
    /// Owned response body.
    body: []u8,

    pub fn deinit(self: *HttpAnswer, alloc: Allocator) void {
        alloc.free(self.body);
        self.* = undefined;
    }
};

const PostOperation = struct {
    alloc: Allocator,
    config: *const Config,
    payload: []const u8,

    pub fn run(self: *PostOperation) !HttpAnswer {
        const auth_header = try self.alloc.print("Bearer {s}", .{self.config.api_key});
        defer secret.zeroAndFree(self.alloc, auth_header);

        var extra_headers: [6]std.http.Header = undefined;
        var extra_len: usize = 0;
        for ([_]std.http.Header{
            .{ .name = "HTTP-Referer", .value = "https://github.com/vercel-labs/fx" },
            .{ .name = "X-Title", .value = "fx" },
            .{ .name = "ai-gateway-protocol-version", .value = "0.0.1" },
            .{ .name = "ai-decision-model-specification-version", .value = "4" },
            .{ .name = "ai-model-id", .value = self.config.model_id },
        }) |header| {
            extra_headers[extra_len] = header;
            extra_len += 1;
        }
        if (self.config.team) |team| {
            if (team.len > 0) {
                extra_headers[extra_len] = .{ .name = gateway_client.vercel_ai_gateway_team_header, .value = team };
                extra_len += 1;
            }
        }

        const buffer = try self.alloc.alloc(u8, max_response_bytes + 1);
        defer self.alloc.free(buffer);
        var response_writer = std.Io.Writer.fixed(buffer);
        const result = gateway_client.fetchOnOpenJob(.{
            .location = .{ .url = self.config.url },
            .method = .POST,
            .payload = self.payload,
            .headers = .{
                .authorization = .{ .override = auth_header },
                .content_type = .{ .override = "application/json" },
                .accept_encoding = .omit,
                .user_agent = .{ .override = gateway_client.user_agent },
            },
            .extra_headers = extra_headers[0..extra_len],
            .response_writer = &response_writer,
            .redirect_behavior = .unhandled,
        }) catch |err| switch (err) {
            error.WriteFailed => return error.DecisionResponseTooLarge,
            else => return err,
        };
        const body = response_writer.buffered();
        if (body.len > max_response_bytes) return error.DecisionResponseTooLarge;
        return .{ .status = result.status, .body = try self.alloc.dupe(u8, body) };
    }
};

/// Serializes `request` as the gateway's decision request body. Caller owns
/// the result.
fn requestBodyAlloc(alloc: Allocator, request: decision.Request) Allocator.Error![]u8 {
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    writeRequestBody(&out.writer, request) catch |err| switch (err) {
        error.WriteFailed => return error.OutOfMemory,
    };
    return out.toOwnedSlice();
}

fn writeRequestBody(writer: *std.Io.Writer, request: decision.Request) std.Io.Writer.Error!void {
    var json: std.json.Stringify = .{ .writer = writer };
    try json.beginObject();

    try json.objectField("stateParts");
    try json.beginArray();
    for (request.state) |part| {
        try json.beginObject();
        try json.objectField("type");
        switch (part) {
            .text => |text| {
                try json.write("text");
                try json.objectField("text");
                try json.write(text);
            },
            .json => |value| {
                try json.write("json");
                try json.objectField("value");
                // Already one valid JSON value: `Request.validate` checked it.
                try json.print("{s}", .{value});
            },
        }
        try json.endObject();
    }
    try json.endArray();

    try json.objectField("questions");
    try json.beginObject();
    for (request.questions) |question| {
        try json.objectField(question.id);
        try json.beginObject();
        try json.objectField("type");
        try json.write(@tagName(question.criteria));
        if (question.instructions) |instructions| {
            try json.objectField("instructions");
            try json.write(instructions);
        }
        switch (question.criteria) {
            .choice => |options| {
                try json.objectField("criteria");
                try json.beginObject();
                for (options) |option| {
                    try json.objectField(option.name);
                    try json.write(option.description);
                }
                try json.endObject();
            },
            .score => |levels| {
                try json.objectField("criteria");
                try json.write(levels);
            },
            .boolean => |criteria| if (criteria.when_true != null or criteria.when_false != null) {
                try json.objectField("criteria");
                try json.beginObject();
                if (criteria.when_true) |description| {
                    try json.objectField("true");
                    try json.write(description);
                }
                if (criteria.when_false) |description| {
                    try json.objectField("false");
                    try json.write(description);
                }
                try json.endObject();
            },
        }
        try json.endObject();
    }
    try json.endObject();

    if (request.max_output_tokens) |limit| {
        try json.objectField("maxOutputTokens");
        try json.write(limit);
    }
    try json.endObject();
}

/// Reads the gateway's decision response. The returned response owns its
/// arena, which is built on `alloc`.
fn parseResponse(alloc: Allocator, body: []const u8) error{ OutOfMemory, Malformed }!decision.ModelResponse {
    var arena = std.heap.ArenaAllocator.init(alloc);
    errdefer arena.deinit();
    const a = arena.allocator();

    const root = std.json.parseFromSliceLeaky(std.json.Value, a, body, .{}) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.Malformed,
    };
    if (root != .object) return error.Malformed;
    const answers_value = root.object.get("answers") orelse return error.Malformed;
    if (answers_value != .object) return error.Malformed;

    const answers = try a.alloc(decision.ModelAnswer, answers_value.object.count());
    var entries = answers_value.object.iterator();
    for (answers) |*answer| {
        const entry = entries.next().?;
        answer.* = .{ .question_id = entry.key_ptr.*, .value = try parseAnswer(a, entry.value_ptr.*) };
    }

    var response: decision.ModelResponse = .{ .answers = answers, .arena = undefined };
    if (root.object.get("model")) |model| {
        if (model == .string) response.model_id = model.string;
    }
    if (root.object.get("rounding")) |rounding| {
        if (rounding != .object) return error.Malformed;
        response.rounding = .{
            .probability_decimals = try decimals(rounding.object.get("probabilityDecimals")),
            .score_decimals = try decimals(rounding.object.get("scoreDecimals")),
        };
    }
    if (root.object.get("usage")) |usage| {
        if (usage == .object) response.usage = .{
            .input_tokens = tokenCount(usage.object.get("inputTokens")),
            .output_tokens = tokenCount(usage.object.get("outputTokens")),
        };
    }
    if (root.object.get("providerMetadata")) |metadata| {
        if (metadata == .object) response.provider_metadata_json = try std.json.Stringify.valueAlloc(a, metadata, .{});
    }
    response.arena = arena;
    return response;
}

fn parseAnswer(arena: Allocator, value: std.json.Value) error{ OutOfMemory, Malformed }!decision.ModelAnswer.Value {
    if (value != .object) return error.Malformed;
    const type_value = value.object.get("type") orelse return error.Malformed;
    if (type_value != .string) return error.Malformed;
    const Kind = enum { choice, score, boolean, refusal };
    switch (std.meta.stringToEnum(Kind, type_value.string) orelse return error.Malformed) {
        .choice => {
            const choice = value.object.get("choice") orelse return error.Malformed;
            if (choice != .string) return error.Malformed;
            return .{ .choice = .{
                .choice = choice.string,
                .probabilities = try parseProbabilities(arena, value.object.get("probabilities")),
            } };
        },
        .score => return .{ .score = .{
            .score = number(value.object.get("score")) orelse return error.Malformed,
            .probabilities = try parseProbabilities(arena, value.object.get("probabilities")),
        } },
        .boolean => return .{ .boolean = .{
            .probability = number(value.object.get("probability")) orelse return error.Malformed,
        } },
        .refusal => return .refusal,
    }
}

fn parseProbabilities(
    arena: Allocator,
    value: ?std.json.Value,
) error{ OutOfMemory, Malformed }!?[]const decision.KeyedProbability {
    const present = value orelse return null;
    if (present != .object) return error.Malformed;
    const probabilities = try arena.alloc(decision.KeyedProbability, present.object.count());
    var entries = present.object.iterator();
    for (probabilities) |*probability| {
        const entry = entries.next().?;
        probability.* = .{
            .key = entry.key_ptr.*,
            .probability = number(entry.value_ptr.*) orelse return error.Malformed,
        };
    }
    return probabilities;
}

fn number(value: ?std.json.Value) ?f64 {
    return switch (value orelse return null) {
        .float => |float| float,
        .integer => |integer| @floatFromInt(integer),
        else => null,
    };
}

fn tokenCount(value: ?std.json.Value) ?u64 {
    return switch (value orelse return null) {
        .integer => |integer| std.math.cast(u64, integer),
        else => null,
    };
}

/// A declared rounding precision. The contract bounds its range.
fn decimals(value: ?std.json.Value) error{Malformed}!?u8 {
    return switch (value orelse return null) {
        .integer => |integer| std.math.cast(u8, integer) orelse error.Malformed,
        else => error.Malformed,
    };
}

const testing = std.testing;

const refund_question = [_]decision.Question{.{ .id = "refund", .instructions = "Is the customer asking for a refund?", .criteria = .{ .boolean = .{} } }};

fn testCall(cancel: *std.atomic.Value(bool)) decision.CallOptions {
    return .{
        .deadline = .fromNow(io_mod.getIo(), .{ .clock = .awake, .raw = .fromMilliseconds(5000) }),
        .cancel_flag = cancel,
    };
}

test "isDecisionModelId tells gateway decision models from language models" {
    try testing.expect(isDecisionModelId("typesafe-ai/jev"));
    try testing.expect(isDecisionModelId("openai/gpt-6-luna-decisions"));
    try testing.expect(!isDecisionModelId("openai/gpt-5.6-luna"));
    try testing.expect(!isDecisionModelId(""));
}

test "the decision endpoint sits beside the gateway chat endpoint" {
    var buf: [128]u8 = undefined;
    try testing.expectEqualStrings(
        "https://ai-gateway.vercel.sh/v4/ai/decision-model",
        urlForChatUrl("https://ai-gateway.vercel.sh/v4/ai/language-model", &buf).?,
    );
    try testing.expect(urlForChatUrl("https://example.test/chat", &buf) == null);
    var small: [8]u8 = undefined;
    try testing.expect(urlForChatUrl("https://ai-gateway.vercel.sh/v4/ai/language-model", &small) == null);
}

test "the request body carries ordered state parts and every question type" {
    const body = try requestBodyAlloc(testing.allocator, .{
        .state = &.{ .{ .text = "Charged \"twice\"." }, .{ .json = "{\"plan\":[\"pro\",1]}" } },
        .questions = &.{
            .{ .id = "team", .instructions = "Which team?", .criteria = .{ .choice = &.{ .{ .name = "billing", .description = "Payments" }, .{ .name = "support" } } } },
            .{ .id = "severity", .criteria = .{ .score = &.{ "Cosmetic", null } } },
            .{ .id = "refund", .criteria = .{ .boolean = .{ .when_true = "Wants money back" } } },
            .{ .id = "angry", .instructions = "Is the customer angry?", .criteria = .{ .boolean = .{} } },
        },
        .max_output_tokens = 64,
    });
    defer testing.allocator.free(body);
    try testing.expectEqualStrings(
        \\{"stateParts":[{"type":"text","text":"Charged \"twice\"."},{"type":"json","value":{"plan":["pro",1]}}],"questions":{"team":{"type":"choice","instructions":"Which team?","criteria":{"billing":"Payments","support":null}},"severity":{"type":"score","criteria":["Cosmetic",null]},"refund":{"type":"boolean","criteria":{"true":"Wants money back"}},"angry":{"type":"boolean","instructions":"Is the customer angry?"}},"maxOutputTokens":64}
    , body);
}

test "a gateway decision posts to the endpoint and decides every question" {
    var server = try FakeGateway.init("200 OK",
        \\{"answers":{"refund":{"type":"boolean","probability":0.98},"team":{"type":"choice","choice":"billing","probabilities":{"billing":0.9,"support":0.1}},"severity":{"type":"score","score":1.3,"probabilities":{"0":0.12,"1":0.46,"2":0.42}}},"model":"openai/gpt-6-luna-decisions","rounding":{"probabilityDecimals":2,"scoreDecimals":2},"usage":{"inputTokens":397,"outputTokens":0},"warnings":[],"providerMetadata":{"openai":{"confidence":{"team":1}}}}
    );
    defer server.deinit();
    try server.start();
    var config = Config{ .api_key = "gw-key", .team = "test-team", .url = server.url, .model_id = "openai/gpt-6-luna-decisions" };
    var cancel = std.atomic.Value(bool).init(false);

    var outcome = try decision.decide(testing.allocator, config.model(), .{
        .state = &.{.{ .text = "I was charged twice." }},
        .questions = &.{
            .{ .id = "team", .criteria = .{ .choice = &.{ .{ .name = "billing" }, .{ .name = "support" } } } },
            .{ .id = "severity", .criteria = .{ .score = &.{ "Cosmetic", "Workaround exists", "Blocking" } } },
            refund_question[0],
        },
    }, testCall(&cancel));
    defer outcome.deinit();

    const result = outcome.decided;
    try testing.expectEqual(@as(usize, 0), result.answers[0].choice.option);
    try testing.expectEqual(@as(f64, 1.3), result.answers[1].score.score);
    try testing.expectEqual(@as(f64, 0.98), result.answers[2].boolean.probability);
    try testing.expectEqual(@as(?u64, 397), result.usage.input_tokens);
    try testing.expectEqualStrings("openai/gpt-6-luna-decisions", result.model_id);
    try testing.expectEqualStrings("{\"openai\":{\"confidence\":{\"team\":1}}}", result.provider_metadata_json.?);

    const sent = server.captured();
    try testing.expect(std.mem.startsWith(u8, sent, "POST /v4/ai/decision-model HTTP"));
    for ([_][]const u8{
        "authorization: Bearer gw-key\r\n",
        "ai-gateway-protocol-version: 0.0.1\r\n",
        "ai-decision-model-specification-version: 4\r\n",
        "ai-model-id: openai/gpt-6-luna-decisions\r\n",
        "x-vercel-ai-gateway-team: test-team\r\n",
        "\r\n\r\n{\"stateParts\":[{\"type\":\"text\",\"text\":\"I was charged twice.\"}],\"questions\":{\"team\":",
    }) |expected| {
        try testing.expect(std.ascii.findIgnoreCase(sent, expected) != null);
    }
}

test "a failed gateway decision reports how it failed" {
    const refusal =
        \\{"answers":{"refund":{"type":"refusal"}}}
    ;
    const cases = [_]struct { status: []const u8, body: []const u8, expected: std.meta.Tag(decision.Outcome) }{
        .{ .status = "500 Internal Server Error", .body = "upstream unavailable", .expected = .transient_failure },
        .{ .status = "429 Too Many Requests", .body = "{}", .expected = .transient_failure },
        .{ .status = "403 Forbidden", .body = "{}", .expected = .permanent_failure },
        .{ .status = "200 OK", .body = "not json", .expected = .invalid_response },
        .{ .status = "200 OK", .body = "{\"answers\":{\"refund\":{\"type\":\"box\"}}}", .expected = .invalid_response },
        .{ .status = "200 OK", .body = "{\"answers\":{\"refund\":{\"type\":\"boolean\",\"probability\":\"high\"}}}", .expected = .invalid_response },
        .{ .status = "200 OK", .body = refusal, .expected = .refused },
    };
    for (cases) |case| {
        var server = try FakeGateway.init(case.status, case.body);
        defer server.deinit();
        try server.start();
        var config = Config{ .api_key = "gw-key", .url = server.url, .model_id = "typesafe-ai/jev" };
        var cancel = std.atomic.Value(bool).init(false);
        var outcome = try decision.decide(testing.allocator, config.model(), .{ .state = &.{}, .questions = &refund_question }, testCall(&cancel));
        defer outcome.deinit();
        try testing.expectEqual(case.expected, std.meta.activeTag(outcome));
    }
}

test "a gateway decision without a credential or after cancellation sends nothing" {
    // Nothing listens here, so a request that went out would fail another way.
    const url = "http://127.0.0.1:1/v4/ai/decision-model";
    const request: decision.Request = .{ .state = &.{}, .questions = &refund_question };

    var unconfigured = Config{ .api_key = "", .url = url, .model_id = "typesafe-ai/jev" };
    var cancel = std.atomic.Value(bool).init(false);
    var missing = try decision.decide(testing.allocator, unconfigured.model(), request, testCall(&cancel));
    defer missing.deinit();
    try testing.expectEqual(std.meta.Tag(decision.Outcome).permanent_failure, std.meta.activeTag(missing));

    var config = Config{ .api_key = "gw-key", .url = url, .model_id = "typesafe-ai/jev" };
    cancel.store(true, .seq_cst);
    var cancelled = try decision.decide(testing.allocator, config.model(), request, testCall(&cancel));
    defer cancelled.deinit();
    try testing.expectEqual(std.meta.Tag(decision.Outcome).cancelled, std.meta.activeTag(cancelled));
}

/// Answers one request with a fixed status and body, and keeps what it read.
const FakeGateway = struct {
    io_backend: std.Io.Threaded = .init_single_threaded,
    server: std.Io.net.Server,
    url: []u8,
    status: []const u8,
    body: []const u8,
    serving: testing_listener.ServerThread = .{},
    failure: ?anyerror = null,
    request: [4096]u8 = undefined,
    request_len: usize = 0,

    fn init(status: []const u8, body: []const u8) !FakeGateway {
        var self = FakeGateway{ .server = undefined, .url = undefined, .status = status, .body = body };
        const address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
        self.server = try address.listen(self.io_backend.io(), .{ .reuse_address = true });
        errdefer self.server.deinit(self.io_backend.io());
        self.url = try testing.allocator.print("http://127.0.0.1:{d}/v4/ai/decision-model", .{self.server.socket.address.getPort()});
        return self;
    }

    fn captured(self: *const FakeGateway) []const u8 {
        return self.request[0..self.request_len];
    }

    fn start(self: *FakeGateway) !void {
        std.debug.assert(self.serving.thread == null);
        self.serving.thread = try std.Thread.spawn(.{}, serveOne, .{self});
    }

    fn stop(self: *FakeGateway, limit_ms: u32) !void {
        try self.serving.stop(&self.server, limit_ms);
    }

    fn deinit(self: *FakeGateway) void {
        self.stop(5_000) catch |err| {
            if (self.failure == null) self.failure = err;
        };
        self.server.deinit(self.io_backend.io());
        testing.allocator.free(self.url);
        if (self.failure) |err| std.debug.panic("fake gateway failed: {s}", .{@errorName(err)});
    }

    fn serveOne(self: *FakeGateway) void {
        defer self.serving.done.store(true, .seq_cst);
        self.serveOneFallible() catch |err| {
            self.failure = err;
        };
    }

    fn serveOneFallible(self: *FakeGateway) !void {
        const zio = self.io_backend.io();
        var stream = (try self.serving.accept(&self.server, zio)) orelse return;
        defer self.serving.close(zio, stream);
        if (self.serving.stopping.load(.seq_cst)) return;

        // Read the head, then the body its content-length announces.
        var socket_buffer: [4096]u8 = undefined;
        var reader = stream.reader(zio, &socket_buffer);
        var body_start: ?usize = null;
        var body_len: usize = 0;
        while (self.request_len < self.request.len) {
            if (body_start) |body_at| {
                if (self.request_len - body_at >= body_len) break;
            }
            self.request[self.request_len] = reader.interface.takeByte() catch |err| switch (err) {
                error.EndOfStream => break,
                else => return err,
            };
            self.request_len += 1;
            const read = self.captured();
            if (body_start == null and std.mem.endsWith(u8, read, "\r\n\r\n")) {
                body_start = read.len;
                const name = "content-length: ";
                const at = std.ascii.findIgnoreCase(read, name) orelse return error.MissingContentLength;
                const end = std.mem.findPos(u8, read, at, "\r\n").?;
                body_len = try std.fmt.parseUnsigned(usize, read[at + name.len .. end], 10);
            }
        }

        var response: [2048]u8 = undefined;
        const written = try std.mem.print(&response, "HTTP/1.1 {s}\r\ncontent-type: application/json\r\ncontent-length: {d}\r\nconnection: close\r\n\r\n{s}", .{ self.status, self.body.len, self.body });
        var write_buffer: [4096]u8 = undefined;
        var writer = stream.writer(zio, &write_buffer);
        try writer.interface.writeAll(written);
        try writer.interface.flush();
    }
};
