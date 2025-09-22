//! zmath - Advanced Mathematics Library for Zig
//! Provides linear algebra, statistics, numerical computing capabilities
//! Replaces functionality from BLAS, LAPACK, GSL, and similar libraries

const std = @import("std");

// Core types
pub const Vector = struct {
    data: []f64,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, len: usize) !Self {
        const data = try allocator.alloc(f64, len);
        @memset(data, 0.0);
        return Self{
            .data = data,
            .allocator = allocator,
        };
    }

    pub fn initWithData(allocator: std.mem.Allocator, values: []const f64) !Self {
        const data = try allocator.alloc(f64, values.len);
        @memcpy(data, values);
        return Self{
            .data = data,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: Self) void {
        self.allocator.free(self.data);
    }

    pub fn size(self: Self) usize {
        return self.data.len;
    }

    pub fn get(self: Self, index: usize) f64 {
        return self.data[index];
    }

    pub fn set(self: Self, index: usize, value: f64) void {
        self.data[index] = value;
    }
};

pub const Matrix = struct {
    data: []f64,
    rows: usize,
    cols: usize,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) !Self {
        const data = try allocator.alloc(f64, rows * cols);
        @memset(data, 0.0);
        return Self{
            .data = data,
            .rows = rows,
            .cols = cols,
            .allocator = allocator,
        };
    }

    pub fn initWithData(allocator: std.mem.Allocator, rows: usize, cols: usize, values: []const f64) !Self {
        if (values.len != rows * cols) {
            return error.InvalidDimensions;
        }
        const data = try allocator.alloc(f64, values.len);
        @memcpy(data, values);
        return Self{
            .data = data,
            .rows = rows,
            .cols = cols,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: Self) void {
        self.allocator.free(self.data);
    }

    pub fn get(self: Self, row: usize, col: usize) f64 {
        return self.data[row * self.cols + col];
    }

    pub fn set(self: Self, row: usize, col: usize, value: f64) void {
        self.data[row * self.cols + col] = value;
    }
};

// Vector operations
pub const vec = struct {
    pub fn add(allocator: std.mem.Allocator, a: Vector, b: Vector) !Vector {
        if (a.size() != b.size()) {
            return error.DimensionMismatch;
        }

        var result = try Vector.init(allocator, a.size());
        for (0..a.size()) |i| {
            result.set(i, a.get(i) + b.get(i));
        }
        return result;
    }

    pub fn subtract(allocator: std.mem.Allocator, a: Vector, b: Vector) !Vector {
        if (a.size() != b.size()) {
            return error.DimensionMismatch;
        }

        var result = try Vector.init(allocator, a.size());
        for (0..a.size()) |i| {
            result.set(i, a.get(i) - b.get(i));
        }
        return result;
    }

    pub fn scalarMultiply(allocator: std.mem.Allocator, a: Vector, scalar: f64) !Vector {
        var result = try Vector.init(allocator, a.size());
        for (0..a.size()) |i| {
            result.set(i, a.get(i) * scalar);
        }
        return result;
    }

    pub fn dot(a: Vector, b: Vector) !f64 {
        if (a.size() != b.size()) {
            return error.DimensionMismatch;
        }

        var sum: f64 = 0.0;
        for (0..a.size()) |i| {
            sum += a.get(i) * b.get(i);
        }
        return sum;
    }

    pub fn magnitude(a: Vector) f64 {
        var sum: f64 = 0.0;
        for (0..a.size()) |i| {
            const val = a.get(i);
            sum += val * val;
        }
        return @sqrt(sum);
    }
};

// Matrix operations
pub const mat = struct {
    pub fn add(allocator: std.mem.Allocator, a: Matrix, b: Matrix) !Matrix {
        if (a.rows != b.rows or a.cols != b.cols) {
            return error.DimensionMismatch;
        }

        var result = try Matrix.init(allocator, a.rows, a.cols);
        for (0..a.rows) |i| {
            for (0..a.cols) |j| {
                result.set(i, j, a.get(i, j) + b.get(i, j));
            }
        }
        return result;
    }

    pub fn subtract(allocator: std.mem.Allocator, a: Matrix, b: Matrix) !Matrix {
        if (a.rows != b.rows or a.cols != b.cols) {
            return error.DimensionMismatch;
        }

        var result = try Matrix.init(allocator, a.rows, a.cols);
        for (0..a.rows) |i| {
            for (0..a.cols) |j| {
                result.set(i, j, a.get(i, j) - b.get(i, j));
            }
        }
        return result;
    }

    pub fn multiply(allocator: std.mem.Allocator, a: Matrix, b: Matrix) !Matrix {
        if (a.cols != b.rows) {
            return error.DimensionMismatch;
        }

        var result = try Matrix.init(allocator, a.rows, b.cols);
        for (0..a.rows) |i| {
            for (0..b.cols) |j| {
                var sum: f64 = 0.0;
                for (0..a.cols) |k| {
                    sum += a.get(i, k) * b.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        return result;
    }

    pub fn transpose(allocator: std.mem.Allocator, a: Matrix) !Matrix {
        var result = try Matrix.init(allocator, a.cols, a.rows);
        for (0..a.rows) |i| {
            for (0..a.cols) |j| {
                result.set(j, i, a.get(i, j));
            }
        }
        return result;
    }

    pub fn scalarMultiply(allocator: std.mem.Allocator, a: Matrix, scalar: f64) !Matrix {
        var result = try Matrix.init(allocator, a.rows, a.cols);
        for (0..a.rows) |i| {
            for (0..a.cols) |j| {
                result.set(i, j, a.get(i, j) * scalar);
            }
        }
        return result;
    }
};

// Statistical functions
pub const stats = struct {
    pub fn mean(data: []const f64) f64 {
        if (data.len == 0) return 0.0;

        var sum: f64 = 0.0;
        for (data) |value| {
            sum += value;
        }
        return sum / @as(f64, @floatFromInt(data.len));
    }

    pub fn variance(data: []const f64) f64 {
        if (data.len == 0) return 0.0;

        const m = mean(data);
        var sum_sq_diff: f64 = 0.0;
        for (data) |value| {
            const diff = value - m;
            sum_sq_diff += diff * diff;
        }
        return sum_sq_diff / @as(f64, @floatFromInt(data.len));
    }

    pub fn standardDeviation(data: []const f64) f64 {
        return @sqrt(variance(data));
    }

    pub fn min(data: []const f64) ?f64 {
        if (data.len == 0) return null;

        var minimum = data[0];
        for (data[1..]) |value| {
            if (value < minimum) minimum = value;
        }
        return minimum;
    }

    pub fn max(data: []const f64) ?f64 {
        if (data.len == 0) return null;

        var maximum = data[0];
        for (data[1..]) |value| {
            if (value > maximum) maximum = value;
        }
        return maximum;
    }
};

// Legacy function for backward compatibility
pub fn bufferedPrint() !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print("zmath - Advanced Mathematics Library for Zig\n", .{});
    try stdout.print("Features: Linear Algebra, Statistics, Numerical Computing\n", .{});

    try stdout.flush();
}

pub fn add(a: i32, b: i32) i32 {
    return a + b;
}

// Tests
test "basic add functionality" {
    try std.testing.expect(add(3, 7) == 10);
}

test "Vector creation and basic operations" {
    const allocator = std.testing.allocator;

    // Test vector creation
    var v1 = try Vector.init(allocator, 3);
    defer v1.deinit();
    v1.set(0, 1.0);
    v1.set(1, 2.0);
    v1.set(2, 3.0);

    try std.testing.expectEqual(@as(usize, 3), v1.size());
    try std.testing.expectEqual(@as(f64, 1.0), v1.get(0));
    try std.testing.expectEqual(@as(f64, 2.0), v1.get(1));
    try std.testing.expectEqual(@as(f64, 3.0), v1.get(2));

    // Test vector creation with data
    const data = [_]f64{ 4.0, 5.0, 6.0 };
    var v2 = try Vector.initWithData(allocator, &data);
    defer v2.deinit();

    try std.testing.expectEqual(@as(f64, 4.0), v2.get(0));
    try std.testing.expectEqual(@as(f64, 5.0), v2.get(1));
    try std.testing.expectEqual(@as(f64, 6.0), v2.get(2));
}

test "Vector arithmetic operations" {
    const allocator = std.testing.allocator;

    const data1 = [_]f64{ 1.0, 2.0, 3.0 };
    const data2 = [_]f64{ 4.0, 5.0, 6.0 };

    var v1 = try Vector.initWithData(allocator, &data1);
    defer v1.deinit();
    var v2 = try Vector.initWithData(allocator, &data2);
    defer v2.deinit();

    // Test vector addition
    var v_add = try vec.add(allocator, v1, v2);
    defer v_add.deinit();
    try std.testing.expectEqual(@as(f64, 5.0), v_add.get(0));
    try std.testing.expectEqual(@as(f64, 7.0), v_add.get(1));
    try std.testing.expectEqual(@as(f64, 9.0), v_add.get(2));

    // Test vector subtraction
    var v_sub = try vec.subtract(allocator, v2, v1);
    defer v_sub.deinit();
    try std.testing.expectEqual(@as(f64, 3.0), v_sub.get(0));
    try std.testing.expectEqual(@as(f64, 3.0), v_sub.get(1));
    try std.testing.expectEqual(@as(f64, 3.0), v_sub.get(2));

    // Test scalar multiplication
    var v_scalar = try vec.scalarMultiply(allocator, v1, 2.0);
    defer v_scalar.deinit();
    try std.testing.expectEqual(@as(f64, 2.0), v_scalar.get(0));
    try std.testing.expectEqual(@as(f64, 4.0), v_scalar.get(1));
    try std.testing.expectEqual(@as(f64, 6.0), v_scalar.get(2));

    // Test dot product
    const dot_result = try vec.dot(v1, v2);
    try std.testing.expectEqual(@as(f64, 32.0), dot_result); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

    // Test magnitude
    const mag = vec.magnitude(v1);
    const expected_mag = @sqrt(1.0 + 4.0 + 9.0); // sqrt(14)
    try std.testing.expectApproxEqAbs(expected_mag, mag, 1e-10);
}

test "Matrix creation and basic operations" {
    const allocator = std.testing.allocator;

    // Test matrix creation
    var m1 = try Matrix.init(allocator, 2, 3);
    defer m1.deinit();
    m1.set(0, 0, 1.0);
    m1.set(0, 1, 2.0);
    m1.set(0, 2, 3.0);
    m1.set(1, 0, 4.0);
    m1.set(1, 1, 5.0);
    m1.set(1, 2, 6.0);

    try std.testing.expectEqual(@as(f64, 1.0), m1.get(0, 0));
    try std.testing.expectEqual(@as(f64, 2.0), m1.get(0, 1));
    try std.testing.expectEqual(@as(f64, 3.0), m1.get(0, 2));
    try std.testing.expectEqual(@as(f64, 4.0), m1.get(1, 0));
    try std.testing.expectEqual(@as(f64, 5.0), m1.get(1, 1));
    try std.testing.expectEqual(@as(f64, 6.0), m1.get(1, 2));

    // Test matrix creation with data
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var m2 = try Matrix.initWithData(allocator, 2, 2, &data);
    defer m2.deinit();

    try std.testing.expectEqual(@as(f64, 1.0), m2.get(0, 0));
    try std.testing.expectEqual(@as(f64, 2.0), m2.get(0, 1));
    try std.testing.expectEqual(@as(f64, 3.0), m2.get(1, 0));
    try std.testing.expectEqual(@as(f64, 4.0), m2.get(1, 1));
}

test "Matrix arithmetic operations" {
    const allocator = std.testing.allocator;

    const data1 = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const data2 = [_]f64{ 5.0, 6.0, 7.0, 8.0 };

    var m1 = try Matrix.initWithData(allocator, 2, 2, &data1);
    defer m1.deinit();
    var m2 = try Matrix.initWithData(allocator, 2, 2, &data2);
    defer m2.deinit();

    // Test matrix addition
    var m_add = try mat.add(allocator, m1, m2);
    defer m_add.deinit();
    try std.testing.expectEqual(@as(f64, 6.0), m_add.get(0, 0));
    try std.testing.expectEqual(@as(f64, 8.0), m_add.get(0, 1));
    try std.testing.expectEqual(@as(f64, 10.0), m_add.get(1, 0));
    try std.testing.expectEqual(@as(f64, 12.0), m_add.get(1, 1));

    // Test matrix subtraction
    var m_sub = try mat.subtract(allocator, m2, m1);
    defer m_sub.deinit();
    try std.testing.expectEqual(@as(f64, 4.0), m_sub.get(0, 0));
    try std.testing.expectEqual(@as(f64, 4.0), m_sub.get(0, 1));
    try std.testing.expectEqual(@as(f64, 4.0), m_sub.get(1, 0));
    try std.testing.expectEqual(@as(f64, 4.0), m_sub.get(1, 1));

    // Test scalar multiplication
    var m_scalar = try mat.scalarMultiply(allocator, m1, 3.0);
    defer m_scalar.deinit();
    try std.testing.expectEqual(@as(f64, 3.0), m_scalar.get(0, 0));
    try std.testing.expectEqual(@as(f64, 6.0), m_scalar.get(0, 1));
    try std.testing.expectEqual(@as(f64, 9.0), m_scalar.get(1, 0));
    try std.testing.expectEqual(@as(f64, 12.0), m_scalar.get(1, 1));
}

test "Matrix multiplication and transpose" {
    const allocator = std.testing.allocator;

    // Matrix multiplication test: [1 2] * [5 6] = [19 22]
    //                              [3 4]   [7 8]   [43 50]
    const data1 = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const data2 = [_]f64{ 5.0, 6.0, 7.0, 8.0 };

    var m1 = try Matrix.initWithData(allocator, 2, 2, &data1);
    defer m1.deinit();
    var m2 = try Matrix.initWithData(allocator, 2, 2, &data2);
    defer m2.deinit();

    var m_mult = try mat.multiply(allocator, m1, m2);
    defer m_mult.deinit();
    try std.testing.expectEqual(@as(f64, 19.0), m_mult.get(0, 0)); // 1*5 + 2*7
    try std.testing.expectEqual(@as(f64, 22.0), m_mult.get(0, 1)); // 1*6 + 2*8
    try std.testing.expectEqual(@as(f64, 43.0), m_mult.get(1, 0)); // 3*5 + 4*7
    try std.testing.expectEqual(@as(f64, 50.0), m_mult.get(1, 1)); // 3*6 + 4*8

    // Test transpose
    var m_trans = try mat.transpose(allocator, m1);
    defer m_trans.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), m_trans.get(0, 0));
    try std.testing.expectEqual(@as(f64, 3.0), m_trans.get(0, 1));
    try std.testing.expectEqual(@as(f64, 2.0), m_trans.get(1, 0));
    try std.testing.expectEqual(@as(f64, 4.0), m_trans.get(1, 1));
}

test "Statistical functions" {
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };

    // Test mean
    const mean_result = stats.mean(&data);
    try std.testing.expectEqual(@as(f64, 3.0), mean_result);

    // Test variance
    const var_result = stats.variance(&data);
    try std.testing.expectEqual(@as(f64, 2.0), var_result); // variance of [1,2,3,4,5] is 2.0

    // Test standard deviation
    const std_result = stats.standardDeviation(&data);
    try std.testing.expectApproxEqAbs(@sqrt(2.0), std_result, 1e-10);

    // Test min and max
    const min_result = stats.min(&data);
    const max_result = stats.max(&data);
    try std.testing.expectEqual(@as(f64, 1.0), min_result.?);
    try std.testing.expectEqual(@as(f64, 5.0), max_result.?);

    // Test empty array
    const empty_data = [_]f64{};
    try std.testing.expectEqual(@as(f64, 0.0), stats.mean(&empty_data));
    try std.testing.expectEqual(@as(f64, 0.0), stats.variance(&empty_data));
    try std.testing.expect(stats.min(&empty_data) == null);
    try std.testing.expect(stats.max(&empty_data) == null);
}

test "Error handling" {
    const allocator = std.testing.allocator;

    // Test dimension mismatch for vectors
    const data1 = [_]f64{ 1.0, 2.0 };
    const data2 = [_]f64{ 3.0, 4.0, 5.0 };

    var v1 = try Vector.initWithData(allocator, &data1);
    defer v1.deinit();
    var v2 = try Vector.initWithData(allocator, &data2);
    defer v2.deinit();

    // This should return an error
    const add_result = vec.add(allocator, v1, v2);
    try std.testing.expectError(error.DimensionMismatch, add_result);

    const dot_result = vec.dot(v1, v2);
    try std.testing.expectError(error.DimensionMismatch, dot_result);

    // Test invalid matrix dimensions
    const matrix_data = [_]f64{ 1.0, 2.0, 3.0 };
    const matrix_result = Matrix.initWithData(allocator, 2, 2, &matrix_data); // 3 elements for 2x2 matrix
    try std.testing.expectError(error.InvalidDimensions, matrix_result);
}
