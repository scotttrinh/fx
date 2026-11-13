//! Decision models answer typed questions about one shared state. Each answer
//! is a choice, a score, or the probability of yes, never generated text.
//!
//! The contract follows the AI SDK's `experimental_decide`
//! (https://ai-sdk.dev/docs/ai-sdk-core/decisions): a caller hands `decide` a
//! `Model`, a state, and questions, and gets back exactly one checked answer
//! for every question or no answers at all. A provider implements `Model`;
//! `decide` rejects a bad request before the provider is called and rejects a
//! provider's answers that break the contract, so every caller reads the same
//! guarantees whichever model answered.
//!
//! fx carries the text and JSON state parts and the choice, score, and
//! boolean questions. File state parts and box questions have no caller yet.

const std = @import("std");
const testing_allocator = @import("../shared/testing_allocator.zig");

const Allocator = std.mem.Allocator;

pub const QuestionType = enum { choice, score, boolean };
pub const QuestionTypes = std.EnumSet(QuestionType);

/// One option of a choice question.
pub const Option = struct {
    name: []const u8,
    /// What choosing the option means. Null sends the name alone.
    description: ?[]const u8 = null,
};

/// One judgment to make about the shared state.
pub const Question = struct {
    /// Names the question to the model. Unique within a request.
    id: []const u8,
    /// What to judge. May be null when the criteria alone define the judgment.
    instructions: ?[]const u8 = null,
    criteria: Criteria,

    pub const Criteria = union(QuestionType) {
        /// The options to choose from. At least one, with unique names.
        choice: []const Option,
        /// Ordered level descriptions, indexed from zero. At least one.
        score: []const ?[]const u8,
        boolean: Boolean,
    };

    /// What yes and no mean. A boolean question without instructions must
    /// describe at least one of them.
    pub const Boolean = struct {
        when_true: ?[]const u8 = null,
        when_false: ?[]const u8 = null,
    };

    fn valid(self: Question) bool {
        if (self.id.len == 0) return false;
        switch (self.criteria) {
            .choice => |options| {
                if (options.len == 0) return false;
                for (options, 0..) |option, index| {
                    if (option.name.len == 0) return false;
                    for (options[0..index]) |earlier| {
                        if (std.mem.eql(u8, earlier.name, option.name)) return false;
                    }
                }
            },
            .score => |levels| if (levels.len == 0) return false,
            .boolean => |criteria| {
                if (self.instructions == null and criteria.when_true == null and criteria.when_false == null) return false;
            },
        }
        return true;
    }
};

/// One ordered piece of the state every question is asked about.
pub const StatePart = union(enum) {
    text: []const u8,
    /// One serialized JSON value.
    json: []const u8,
};

/// Borrowed for the length of one `decide` call.
pub const Request = struct {
    state: []const StatePart,
    questions: []const Question,
    max_output_tokens: ?u32 = null,

    /// Checks everything a provider would otherwise have to reject: at least
    /// one question, unique question ids, well-formed criteria, and JSON
    /// state parts that parse.
    pub fn validate(self: Request, alloc: Allocator) (Allocator.Error || error{InvalidRequest})!void {
        if (self.questions.len == 0) return error.InvalidRequest;
        for (self.questions, 0..) |question, index| {
            if (!question.valid()) return error.InvalidRequest;
            for (self.questions[0..index]) |earlier| {
                if (std.mem.eql(u8, earlier.id, question.id)) return error.InvalidRequest;
            }
        }
        for (self.state) |part| switch (part) {
            .text => {},
            .json => |value| if (!try std.json.validate(alloc, value)) return error.InvalidRequest,
        };
    }
};

pub const Answer = union(QuestionType) {
    choice: Choice,
    score: Score,
    boolean: Boolean,

    pub const Choice = struct {
        /// Index of the chosen option in the question's criteria.
        option: usize,
        /// One probability per option, in criteria order, when the model
        /// reports a distribution. The chosen option's is the highest.
        probabilities: ?[]const f64 = null,
    };

    pub const Score = struct {
        /// Position on the rubric, from zero to the last level's index.
        score: f64,
        /// One probability per level, in order, when the model reports a
        /// distribution. `score` is its probability-weighted mean.
        probabilities: ?[]const f64 = null,
    };

    pub const Boolean = struct {
        /// The model's estimate that the answer is yes. Not its confidence in
        /// either outcome: 0.02 is a strong no.
        probability: f64,
    };
};

pub const Usage = struct {
    input_tokens: ?u64 = null,
    output_tokens: ?u64 = null,
};

/// Decimal places a provider rounds its output to. `decide` widens its checks
/// of distribution sums and weighted scores by half a unit in the last place
/// for each rounded value, and never changes the values.
pub const Rounding = struct {
    probability_decimals: ?u8 = null,
    score_decimals: ?u8 = null,
};

/// Owns every slice it holds. Call `deinit` when done.
pub const Result = struct {
    /// One answer per question, in the request's question order.
    answers: []const Answer,
    usage: Usage = .{},
    /// The model that answered, as the provider reported it.
    model_id: []const u8,
    /// The provider's own serialized metadata object, such as a confidence
    /// statistic. Its shape belongs to the provider.
    provider_metadata_json: ?[]const u8 = null,
    arena: std.heap.ArenaAllocator,

    pub fn deinit(self: *Result) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub const Outcome = union(enum) {
    decided: Result,
    /// The model declined at least one question. Asking again does not help.
    refused,
    /// The model's answers broke the contract: an answer missing, extra, or
    /// of the wrong type, an unknown option, a score off the rubric, or
    /// probabilities that are not a distribution.
    invalid_response,
    transient_failure,
    permanent_failure,
    timed_out,
    cancelled,

    pub fn deinit(self: *Outcome) void {
        switch (self.*) {
            .decided => |*result| result.deinit(),
            else => {},
        }
        self.* = undefined;
    }
};

/// The deadline and cancellation one model call runs under. A caller that
/// retries gives each attempt its own.
pub const CallOptions = struct {
    deadline: std.Io.Clock.Timestamp,
    cancel_flag: *std.atomic.Value(bool),
};

/// One probability a provider reported, under the provider's own key: an
/// option name for a choice, a level index such as "2" for a score.
pub const KeyedProbability = struct {
    key: []const u8,
    probability: f64,
};

/// One answer exactly as a provider returned it, before `decide` checks it.
pub const ModelAnswer = struct {
    question_id: []const u8,
    value: Value,

    pub const Value = union(enum) {
        choice: struct { choice: []const u8, probabilities: ?[]const KeyedProbability = null },
        score: struct { score: f64, probabilities: ?[]const KeyedProbability = null },
        boolean: struct { probability: f64 },
        refusal,
    };
};

/// A provider's answers. Every slice is allocated from `arena`, which the
/// receiver owns.
pub const ModelResponse = struct {
    answers: []const ModelAnswer,
    rounding: Rounding = .{},
    usage: Usage = .{},
    model_id: ?[]const u8 = null,
    provider_metadata_json: ?[]const u8 = null,
    arena: std.heap.ArenaAllocator,
};

pub const ModelOutcome = union(enum) {
    response: ModelResponse,
    /// The provider answered with something other than a decision response.
    malformed_response,
    transient_failure,
    permanent_failure,
    timed_out,
    cancelled,
};

/// A provider's decision model. `context` is borrowed for every call.
pub const Model = struct {
    provider: []const u8,
    model_id: []const u8,
    /// `decide` rejects a question of any other type before calling
    /// `decide_fn`.
    supported_question_types: QuestionTypes,
    context: *anyopaque,
    /// Answers every question against the same state in one call, with no
    /// partial result. `request` has already passed `Request.validate`.
    decide_fn: *const fn (
        context: *anyopaque,
        alloc: Allocator,
        request: Request,
        call: CallOptions,
    ) Allocator.Error!ModelOutcome,
};

pub const DecideError = Allocator.Error || error{
    /// The request breaks `Request.validate`.
    InvalidRequest,
    /// The model does not answer one of the request's question types.
    UnsupportedQuestionType,
};

/// Asks `model` every question in `request` once. The caller owns the
/// returned outcome and retries a failed call if it wants to; `decide` makes
/// one attempt and never substitutes another model.
pub fn decide(alloc: Allocator, model: Model, request: Request, call: CallOptions) DecideError!Outcome {
    try request.validate(alloc);
    for (request.questions) |question| {
        if (!model.supported_question_types.contains(std.meta.activeTag(question.criteria))) return error.UnsupportedQuestionType;
    }

    var response = switch (try model.decide_fn(model.context, alloc, request, call)) {
        .response => |response| response,
        .malformed_response => return .invalid_response,
        .transient_failure => return .transient_failure,
        .permanent_failure => return .permanent_failure,
        .timed_out => return .timed_out,
        .cancelled => return .cancelled,
    };
    errdefer response.arena.deinit();

    const arena = response.arena.allocator();
    const answers = switch (try checkedAnswers(arena, request.questions, response)) {
        .answers => |answers| answers,
        .refused => {
            response.arena.deinit();
            return .refused;
        },
        .invalid => {
            response.arena.deinit();
            return .invalid_response;
        },
    };
    const model_id = response.model_id orelse try arena.dupe(u8, model.model_id);
    return .{ .decided = .{
        .answers = answers,
        .usage = response.usage,
        .model_id = model_id,
        .provider_metadata_json = response.provider_metadata_json,
        .arena = response.arena,
    } };
}

/// Sums and means may be off by this much before rounding is considered.
const tolerance: f64 = 1e-6;
const max_rounding_decimals: u8 = 15;

const Checked = union(enum) {
    answers: []const Answer,
    refused,
    invalid,
};

fn checkedAnswers(
    arena: Allocator,
    questions: []const Question,
    response: ModelResponse,
) Allocator.Error!Checked {
    const probability_error = roundingError(response.rounding.probability_decimals) orelse return .invalid;
    const score_error = roundingError(response.rounding.score_decimals) orelse return .invalid;
    if (response.answers.len != questions.len) return .invalid;

    const answers = try arena.alloc(Answer, questions.len);
    var refused = false;
    for (questions, answers) |question, *answer| {
        // Equal counts and one match per question leave no extra answer.
        const given = for (response.answers) |candidate| {
            if (std.mem.eql(u8, candidate.question_id, question.id)) break candidate.value;
        } else return .invalid;
        if (given == .refusal) {
            refused = true;
            continue;
        }
        answer.* = (try checkedAnswer(arena, question, given, probability_error, score_error)) orelse return .invalid;
    }
    return if (refused) .refused else .{ .answers = answers };
}

/// Null when `given` is not a valid answer to `question`.
fn checkedAnswer(
    arena: Allocator,
    question: Question,
    given: ModelAnswer.Value,
    probability_error: f64,
    score_error: f64,
) Allocator.Error!?Answer {
    switch (question.criteria) {
        .choice => |options| {
            if (given != .choice) return null;
            const selected = optionIndex(options, given.choice.choice) orelse return null;
            const reported = given.choice.probabilities orelse return .{ .choice = .{ .option = selected } };
            const probabilities = try arena.alloc(f64, options.len);
            if (!fillDistribution(probabilities, reported, options, probability_error)) return null;
            for (probabilities) |probability| {
                if (probability > probabilities[selected] + tolerance + 2 * probability_error) return null;
            }
            return .{ .choice = .{ .option = selected, .probabilities = probabilities } };
        },
        .score => |levels| {
            if (given != .score) return null;
            const score = given.score.score;
            const last_level: f64 = @floatFromInt(levels.len - 1);
            if (!std.math.isFinite(score) or score < 0 or score > last_level) return null;
            const reported = given.score.probabilities orelse return .{ .score = .{ .score = score } };
            const probabilities = try arena.alloc(f64, levels.len);
            if (!fillDistribution(probabilities, reported, levels, probability_error)) return null;
            var mean: f64 = 0;
            var mean_error: f64 = 0;
            for (probabilities, 0..) |probability, level| {
                const weight: f64 = @floatFromInt(level);
                mean += weight * probability;
                mean_error += weight * probability_error;
            }
            if (@abs(mean - score) > tolerance + mean_error + score_error) return null;
            return .{ .score = .{ .score = score, .probabilities = probabilities } };
        },
        .boolean => {
            if (given != .boolean or !isProbability(given.boolean.probability)) return null;
            return .{ .boolean = .{ .probability = given.boolean.probability } };
        },
    }
}

/// Half a unit in the last rounded place, zero for full precision, or null
/// when the provider declared more places than a double holds.
fn roundingError(decimals: ?u8) ?f64 {
    const places = decimals orelse return 0;
    if (places > max_rounding_decimals) return null;
    return 0.5 * std.math.pow(f64, 10, -@as(f64, @floatFromInt(places)));
}

fn isProbability(value: f64) bool {
    return std.math.isFinite(value) and value >= 0 and value <= 1;
}

fn optionIndex(options: []const Option, name: []const u8) ?usize {
    for (options, 0..) |option, index| {
        if (std.mem.eql(u8, option.name, name)) return index;
    }
    return null;
}

/// The position `key` names among `keys`: an option by its name, a score
/// level by its index written in decimal.
fn keyIndex(keys: anytype, key: []const u8) ?usize {
    if (@TypeOf(keys) == []const Option) return optionIndex(keys, key);
    var buffer: [20]u8 = undefined;
    for (0..keys.len) |index| {
        const written = std.mem.print(&buffer, "{d}", .{index}) catch unreachable;
        if (std.mem.eql(u8, written, key)) return index;
    }
    return null;
}

/// Copies `reported` into `out` in `keys` order. False unless it gives every
/// key exactly one probability and the probabilities sum to one.
fn fillDistribution(out: []f64, reported: []const KeyedProbability, keys: anytype, rounding_error: f64) bool {
    if (reported.len != out.len) return false;
    @memset(out, std.math.nan(f64));
    var sum: f64 = 0;
    for (reported) |entry| {
        const index = keyIndex(keys, entry.key) orelse return false;
        if (!std.math.isNan(out[index]) or !isProbability(entry.probability)) return false;
        out[index] = entry.probability;
        sum += entry.probability;
    }
    const count: f64 = @floatFromInt(out.len);
    return @abs(sum - 1) <= tolerance + count * rounding_error;
}

const testing = std.testing;

/// A model that returns the answers a test hands it and counts its calls.
const FakeModel = struct {
    answers: []const ModelAnswer = &.{},
    rounding: Rounding = .{},
    failure: ?ModelOutcome = null,
    supported: QuestionTypes = .full,
    calls: usize = 0,

    fn model(self: *FakeModel) Model {
        return .{
            .provider = "fake",
            .model_id = "fake/decider",
            .supported_question_types = self.supported,
            .context = self,
            .decide_fn = decideFake,
        };
    }

    fn decideFake(raw: *anyopaque, alloc: Allocator, _: Request, _: CallOptions) Allocator.Error!ModelOutcome {
        const self: *FakeModel = @ptrCast(@alignCast(raw));
        self.calls += 1;
        if (self.failure) |failure| return failure;
        var arena = std.heap.ArenaAllocator.init(alloc);
        errdefer arena.deinit();
        const answers = try arena.allocator().dupe(ModelAnswer, self.answers);
        return .{ .response = .{
            .answers = answers,
            .rounding = self.rounding,
            .usage = .{ .input_tokens = 12, .output_tokens = 0 },
            .arena = arena,
        } };
    }

    fn decideWith(self: *FakeModel, request: Request) DecideError!Outcome {
        var cancel = std.atomic.Value(bool).init(false);
        return decide(testing.allocator, self.model(), request, .{
            .deadline = .fromNow(testing.io, .{ .clock = .awake, .raw = .fromMilliseconds(1000) }),
            .cancel_flag = &cancel,
        });
    }
};

const triage_questions = [_]Question{
    .{ .id = "team", .instructions = "Which team should handle this?", .criteria = .{ .choice = &.{
        .{ .name = "billing", .description = "Payments and refunds" },
        .{ .name = "support" },
    } } },
    .{ .id = "severity", .instructions = "How severe is the issue?", .criteria = .{ .score = &.{ "Cosmetic", "Workaround exists", null } } },
    .{ .id = "refund", .criteria = .{ .boolean = .{ .when_true = "The customer wants money back" } } },
};
const triage_state = [_]StatePart{
    .{ .text = "I was charged twice." },
    .{ .json = "{\"plan\":\"pro\"}" },
};
const triage: Request = .{ .state = &triage_state, .questions = &triage_questions };

test "decide returns one typed answer per question in question order" {
    // Answers and distribution entries arrive in the provider's own order.
    var fake = FakeModel{ .answers = &.{
        .{ .question_id = "refund", .value = .{ .boolean = .{ .probability = 0.98 } } },
        .{ .question_id = "severity", .value = .{ .score = .{ .score = 1.3, .probabilities = &.{
            .{ .key = "2", .probability = 0.42 },
            .{ .key = "0", .probability = 0.12 },
            .{ .key = "1", .probability = 0.46 },
        } } } },
        .{ .question_id = "team", .value = .{ .choice = .{ .choice = "billing", .probabilities = &.{
            .{ .key = "support", .probability = 0.1 },
            .{ .key = "billing", .probability = 0.9 },
        } } } },
    } };
    var outcome = try fake.decideWith(triage);
    defer outcome.deinit();

    const result = outcome.decided;
    try testing.expectEqual(@as(usize, 0), result.answers[0].choice.option);
    try testing.expectEqualSlices(f64, &.{ 0.9, 0.1 }, result.answers[0].choice.probabilities.?);
    try testing.expectEqual(@as(f64, 1.3), result.answers[1].score.score);
    try testing.expectEqualSlices(f64, &.{ 0.12, 0.46, 0.42 }, result.answers[1].score.probabilities.?);
    try testing.expectEqual(@as(f64, 0.98), result.answers[2].boolean.probability);
    try testing.expectEqual(@as(?u64, 12), result.usage.input_tokens);
    try testing.expectEqualStrings("fake/decider", result.model_id);
}

test "decide rejects a bad request before calling the model" {
    const choice: Question.Criteria = .{ .choice = &.{.{ .name = "yes" }} };
    const cases = [_]struct { request: Request, supported: QuestionTypes = .full, err: DecideError }{
        .{ .request = .{ .state = &.{}, .questions = &.{} }, .err = error.InvalidRequest },
        .{ .request = .{ .state = &.{}, .questions = &.{.{ .id = "", .criteria = choice }} }, .err = error.InvalidRequest },
        .{ .request = .{ .state = &.{}, .questions = &.{ .{ .id = "a", .criteria = choice }, .{ .id = "a", .criteria = choice } } }, .err = error.InvalidRequest },
        .{ .request = .{ .state = &.{}, .questions = &.{.{ .id = "a", .criteria = .{ .choice = &.{} } }} }, .err = error.InvalidRequest },
        .{ .request = .{ .state = &.{}, .questions = &.{.{ .id = "a", .criteria = .{ .choice = &.{ .{ .name = "x" }, .{ .name = "x" } } } }} }, .err = error.InvalidRequest },
        .{ .request = .{ .state = &.{}, .questions = &.{.{ .id = "a", .criteria = .{ .score = &.{} } }} }, .err = error.InvalidRequest },
        .{ .request = .{ .state = &.{}, .questions = &.{.{ .id = "a", .criteria = .{ .boolean = .{} } }} }, .err = error.InvalidRequest },
        .{ .request = .{ .state = &.{.{ .json = "{\"open\":" }}, .questions = &.{.{ .id = "a", .criteria = choice }} }, .err = error.InvalidRequest },
        .{ .request = triage, .supported = .initMany(&.{ .choice, .boolean }), .err = error.UnsupportedQuestionType },
    };
    for (cases) |case| {
        var fake = FakeModel{ .supported = case.supported };
        try testing.expectError(case.err, fake.decideWith(case.request));
        try testing.expectEqual(@as(usize, 0), fake.calls);
    }
}

test "decide rejects answers that break the contract" {
    const Case = struct { question: Question, answers: []const ModelAnswer, rounding: Rounding = .{} };
    const team = triage_questions[0];
    const severity = triage_questions[1];
    const refund = triage_questions[2];
    const cases = [_]Case{
        // Missing, extra, and mistyped answers.
        .{ .question = refund, .answers = &.{} },
        .{ .question = refund, .answers = &.{.{ .question_id = "other", .value = .{ .boolean = .{ .probability = 0.5 } } }} },
        .{ .question = refund, .answers = &.{
            .{ .question_id = "refund", .value = .{ .boolean = .{ .probability = 0.5 } } },
            .{ .question_id = "other", .value = .{ .boolean = .{ .probability = 0.5 } } },
        } },
        .{ .question = refund, .answers = &.{.{ .question_id = "refund", .value = .{ .score = .{ .score = 0 } } }} },
        .{ .question = refund, .answers = &.{.{ .question_id = "refund", .value = .{ .boolean = .{ .probability = 1.2 } } }} },
        // A choice names an option and, with a distribution, the likeliest one.
        .{ .question = team, .answers = &.{.{ .question_id = "team", .value = .{ .choice = .{ .choice = "sales" } } }} },
        .{ .question = team, .answers = &.{.{ .question_id = "team", .value = .{ .choice = .{ .choice = "billing", .probabilities = &.{.{ .key = "billing", .probability = 1 }} } } }} },
        .{ .question = team, .answers = &.{.{ .question_id = "team", .value = .{ .choice = .{ .choice = "billing", .probabilities = &.{ .{ .key = "billing", .probability = 0.5 }, .{ .key = "billing", .probability = 0.5 } } } } }} },
        .{ .question = team, .answers = &.{.{ .question_id = "team", .value = .{ .choice = .{ .choice = "billing", .probabilities = &.{ .{ .key = "billing", .probability = 0.6 }, .{ .key = "support", .probability = 0.3 } } } } }} },
        .{ .question = team, .answers = &.{.{ .question_id = "team", .value = .{ .choice = .{ .choice = "billing", .probabilities = &.{ .{ .key = "billing", .probability = 0.4 }, .{ .key = "support", .probability = 0.6 } } } } }} },
        // A score stays on the rubric and matches its distribution's mean.
        .{ .question = severity, .answers = &.{.{ .question_id = "severity", .value = .{ .score = .{ .score = 2.5 } } }} },
        .{ .question = severity, .answers = &.{.{ .question_id = "severity", .value = .{ .score = .{ .score = 2, .probabilities = &.{ .{ .key = "0", .probability = 0.5 }, .{ .key = "1", .probability = 0.5 }, .{ .key = "2", .probability = 0 } } } } }} },
        .{ .question = severity, .answers = &.{.{ .question_id = "severity", .value = .{ .score = .{ .score = 1, .probabilities = &.{ .{ .key = "0", .probability = 0 }, .{ .key = "01", .probability = 1 }, .{ .key = "2", .probability = 0 } } } } }} },
        // Rounding cannot be finer than a double.
        .{ .question = refund, .answers = &.{.{ .question_id = "refund", .value = .{ .boolean = .{ .probability = 0.5 } } }}, .rounding = .{ .probability_decimals = 16 } },
    };
    for (cases) |case| {
        var fake = FakeModel{ .answers = case.answers, .rounding = case.rounding };
        var outcome = try fake.decideWith(.{ .state = &.{}, .questions = &.{case.question} });
        defer outcome.deinit();
        try testing.expectEqual(std.meta.Tag(Outcome).invalid_response, std.meta.activeTag(outcome));
    }
}

test "declared rounding widens the distribution and weighted-score checks" {
    // Two-decimal rounding leaves this sum at 0.99 and the mean 0.01 off.
    const answers = [_]ModelAnswer{
        .{ .question_id = "team", .value = .{ .choice = .{ .choice = "billing", .probabilities = &.{ .{ .key = "billing", .probability = 0.66 }, .{ .key = "support", .probability = 0.33 } } } } },
        .{ .question_id = "severity", .value = .{ .score = .{ .score = 1.01, .probabilities = &.{ .{ .key = "0", .probability = 0.33 }, .{ .key = "1", .probability = 0.33 }, .{ .key = "2", .probability = 0.33 } } } } },
    };
    const request: Request = .{ .state = &.{}, .questions = triage_questions[0..2] };
    const cases = [_]struct { rounding: Rounding, expected: std.meta.Tag(Outcome) }{
        .{ .rounding = .{ .probability_decimals = 2, .score_decimals = 2 }, .expected = .decided },
        .{ .rounding = .{}, .expected = .invalid_response },
    };
    for (cases) |case| {
        var fake = FakeModel{ .answers = &answers, .rounding = case.rounding };
        var outcome = try fake.decideWith(request);
        defer outcome.deinit();
        try testing.expectEqual(case.expected, std.meta.activeTag(outcome));
        // The provider's values are kept as returned, never renormalized.
        if (outcome == .decided) try testing.expectEqualSlices(f64, &.{ 0.66, 0.33 }, outcome.decided.answers[0].choice.probabilities.?);
    }
}

test "a refused question refuses the whole decision" {
    var fake = FakeModel{ .answers = &.{
        .{ .question_id = "team", .value = .{ .choice = .{ .choice = "billing" } } },
        .{ .question_id = "severity", .value = .refusal },
        .{ .question_id = "refund", .value = .{ .boolean = .{ .probability = 0.5 } } },
    } };
    var outcome = try fake.decideWith(triage);
    defer outcome.deinit();
    try testing.expectEqual(std.meta.Tag(Outcome).refused, std.meta.activeTag(outcome));
}

test "decide reports a failed model call once without retrying" {
    const cases = [_]struct { failure: ModelOutcome, expected: std.meta.Tag(Outcome) }{
        .{ .failure = .malformed_response, .expected = .invalid_response },
        .{ .failure = .transient_failure, .expected = .transient_failure },
        .{ .failure = .permanent_failure, .expected = .permanent_failure },
        .{ .failure = .timed_out, .expected = .timed_out },
        .{ .failure = .cancelled, .expected = .cancelled },
    };
    for (cases) |case| {
        var fake = FakeModel{ .failure = case.failure };
        var outcome = try fake.decideWith(triage);
        defer outcome.deinit();
        try testing.expectEqual(case.expected, std.meta.activeTag(outcome));
        try testing.expectEqual(@as(usize, 1), fake.calls);
    }
}

test "decide frees a partial result when allocation fails" {
    const Run = struct {
        fn run(alloc: Allocator) !void {
            var fake = FakeModel{ .answers = &.{
                .{ .question_id = "team", .value = .{ .choice = .{ .choice = "billing", .probabilities = &.{ .{ .key = "billing", .probability = 0.9 }, .{ .key = "support", .probability = 0.1 } } } } },
                .{ .question_id = "severity", .value = .{ .score = .{ .score = 0 } } },
                .{ .question_id = "refund", .value = .{ .boolean = .{ .probability = 0.98 } } },
            } };
            var cancel = std.atomic.Value(bool).init(false);
            var outcome = try decide(alloc, fake.model(), triage, .{
                .deadline = .fromNow(testing.io, .{ .clock = .awake, .raw = .fromMilliseconds(1000) }),
                .cancel_flag = &cancel,
            });
            outcome.deinit();
        }
    };
    try testing.checkAllAllocationFailures(testing_allocator.no_resize, Run.run, .{});
}
