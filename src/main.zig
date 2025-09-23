const std = @import("std");
const zmath = @import("zmath");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try zmath.bufferedPrint();

    std.debug.print("\n=== zmath Library Demonstration ===\n\n", .{});

    // Demonstrate vector operations
    std.debug.print("1. Vector Operations:\n", .{});

    const v1_data = [_]f64{ 1.0, 2.0, 3.0 };
    const v2_data = [_]f64{ 4.0, 5.0, 6.0 };

    var v1 = try zmath.Vector.initWithData(allocator, &v1_data);
    defer v1.deinit();
    var v2 = try zmath.Vector.initWithData(allocator, &v2_data);
    defer v2.deinit();

    std.debug.print("   Vector v1: [{d:.1}, {d:.1}, {d:.1}]\n", .{ v1.get(0), v1.get(1), v1.get(2) });
    std.debug.print("   Vector v2: [{d:.1}, {d:.1}, {d:.1}]\n", .{ v2.get(0), v2.get(1), v2.get(2) });

    var v_add = try zmath.vec.add(allocator, v1, v2);
    defer v_add.deinit();
    std.debug.print("   v1 + v2:   [{d:.1}, {d:.1}, {d:.1}]\n", .{ v_add.get(0), v_add.get(1), v_add.get(2) });

    const dot_product = try zmath.vec.dot(v1, v2);
    std.debug.print("   v1 · v2:   {d:.1}\n", .{dot_product});

    const magnitude = zmath.vec.magnitude(v1);
    std.debug.print("   |v1|:      {d:.3}\n", .{magnitude});

    // Demonstrate matrix operations
    std.debug.print("\n2. Matrix Operations:\n", .{});

    const m1_data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const m2_data = [_]f64{ 5.0, 6.0, 7.0, 8.0 };

    var m1 = try zmath.Matrix.initWithData(allocator, 2, 2, &m1_data);
    defer m1.deinit();
    var m2 = try zmath.Matrix.initWithData(allocator, 2, 2, &m2_data);
    defer m2.deinit();

    std.debug.print("   Matrix m1:\n", .{});
    std.debug.print("   [{d:.1} {d:.1}]\n", .{ m1.get(0, 0), m1.get(0, 1) });
    std.debug.print("   [{d:.1} {d:.1}]\n", .{ m1.get(1, 0), m1.get(1, 1) });

    std.debug.print("   Matrix m2:\n", .{});
    std.debug.print("   [{d:.1} {d:.1}]\n", .{ m2.get(0, 0), m2.get(0, 1) });
    std.debug.print("   [{d:.1} {d:.1}]\n", .{ m2.get(1, 0), m2.get(1, 1) });

    var m_mult = try zmath.mat.multiply(allocator, m1, m2);
    defer m_mult.deinit();
    std.debug.print("   m1 * m2:\n", .{});
    std.debug.print("   [{d:.1} {d:.1}]\n", .{ m_mult.get(0, 0), m_mult.get(0, 1) });
    std.debug.print("   [{d:.1} {d:.1}]\n", .{ m_mult.get(1, 0), m_mult.get(1, 1) });

    var m_trans = try zmath.mat.transpose(allocator, m1);
    defer m_trans.deinit();
    std.debug.print("   m1^T:\n", .{});
    std.debug.print("   [{d:.1} {d:.1}]\n", .{ m_trans.get(0, 0), m_trans.get(0, 1) });
    std.debug.print("   [{d:.1} {d:.1}]\n", .{ m_trans.get(1, 0), m_trans.get(1, 1) });

    // Demonstrate LU decomposition
    var lu = try zmath.mat.luDecompose(allocator, m1);
    defer lu.l.deinit();
    defer lu.u.deinit();
    std.debug.print("   m1 = L * U:\n", .{});
    std.debug.print("   L (lower triangular):\n", .{});
    std.debug.print("   [{d:.1} {d:.1}]\n", .{ lu.l.get(0, 0), lu.l.get(0, 1) });
    std.debug.print("   [{d:.1} {d:.1}]\n", .{ lu.l.get(1, 0), lu.l.get(1, 1) });
    std.debug.print("   U (upper triangular):\n", .{});
    std.debug.print("   [{d:.1} {d:.1}]\n", .{ lu.u.get(0, 0), lu.u.get(0, 1) });
    std.debug.print("   [{d:.1} {d:.1}]\n", .{ lu.u.get(1, 0), lu.u.get(1, 1) });

    // Demonstrate statistical functions
    std.debug.print("\n3. Statistical Functions:\n", .{});

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };

    std.debug.print("   Data: [", .{});
    for (data, 0..) |value, i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{d:.1}", .{value});
    }
    std.debug.print("]\n", .{});

    const mean = zmath.stats.mean(&data);
    const variance = zmath.stats.variance(&data);
    const std_dev = zmath.stats.standardDeviation(&data);
    const min_val = zmath.stats.min(&data).?;
    const max_val = zmath.stats.max(&data).?;

    std.debug.print("   Mean:              {d:.2}\n", .{mean});
    std.debug.print("   Variance:          {d:.2}\n", .{variance});
    std.debug.print("   Standard Deviation: {d:.3}\n", .{std_dev});
    std.debug.print("   Minimum:           {d:.1}\n", .{min_val});
    std.debug.print("   Maximum:           {d:.1}\n", .{max_val});

    std.debug.print("\n=== End Demonstration ===\n", .{});
}

test "simple test" {
    const gpa = std.testing.allocator;
    var list: std.ArrayList(i32) = .empty;
    defer list.deinit(gpa);
    try list.append(gpa, 42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}

test "fuzz example" {
    const Context = struct {
        fn testOne(context: @This(), input: []const u8) anyerror!void {
            _ = context;
            try std.testing.expect(!std.mem.eql(u8, "canyoufindme", input));
        }
    };
    try std.testing.fuzz(Context{}, Context.testOne, .{});
}
