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

    /// Performs LU decomposition using Doolittle's algorithm.
    /// Decomposes matrix A into L (lower triangular with 1s on diagonal)
    /// and U (upper triangular) such that A = L * U.
    /// Returns a struct containing L and U matrices.
    pub fn luDecompose(allocator: std.mem.Allocator, a: Matrix) !struct { l: Matrix, u: Matrix } {
        if (a.rows != a.cols) {
            return error.NotSquareMatrix;
        }

        const n = a.rows;
        var l = try Matrix.init(allocator, n, n);
        var u = try Matrix.init(allocator, n, n);

        // Initialize L with identity matrix (1s on diagonal)
        for (0..n) |i| {
            l.set(i, i, 1.0);
        }

        // Doolittle's algorithm
        for (0..n) |i| {
            // Upper triangular matrix U
            for (0..n) |k| {
                if (i <= k) {
                    var sum: f64 = 0.0;
                    for (0..i) |j| {
                        sum += l.get(i, j) * u.get(j, k);
                    }
                    u.set(i, k, a.get(i, k) - sum);
                }
            }

            // Lower triangular matrix L
            for (0..n) |k| {
                if (i < k) {
                    var sum: f64 = 0.0;
                    for (0..i) |j| {
                        sum += l.get(k, j) * u.get(j, i);
                    }
                    l.set(k, i, (a.get(k, i) - sum) / u.get(i, i));
                }
            }
        }

        return .{ .l = l, .u = u };
    }

    /// Performs QR decomposition using Gram-Schmidt process.
    /// Decomposes matrix A into Q (orthogonal) and R (upper triangular)
    /// such that A = Q * R.
    /// Returns a struct containing Q and R matrices.
    pub fn qrDecompose(allocator: std.mem.Allocator, a: Matrix) !struct { q: Matrix, r: Matrix } {
        const m = a.rows;
        const n = a.cols;

        var q = try Matrix.init(allocator, m, n);
        var r = try Matrix.init(allocator, n, n);

        // Gram-Schmidt process
        for (0..n) |j| {
            // Start with the j-th column of A
            for (0..m) |i| {
                q.set(i, j, a.get(i, j));
            }

            // Subtract projections onto previous Q columns
            for (0..j) |k| {
                // Compute dot product of q_j with q_k
                var dot_qj_qk: f64 = 0.0;
                for (0..m) |i| {
                    dot_qj_qk += q.get(i, j) * q.get(i, k);
                }

                // Store in R
                r.set(k, j, dot_qj_qk);

                // Subtract projection: q_j = q_j - (dot_qj_qk) * q_k
                for (0..m) |i| {
                    const new_val = q.get(i, j) - dot_qj_qk * q.get(i, k);
                    q.set(i, j, new_val);
                }
            }

            // Normalize q_j and store norm in R
            var norm: f64 = 0.0;
            for (0..m) |i| {
                const val = q.get(i, j);
                norm += val * val;
            }
            norm = @sqrt(norm);

            if (norm > 1e-10) { // Avoid division by zero
                r.set(j, j, norm);
                // Normalize q_j
                for (0..m) |i| {
                    q.set(i, j, q.get(i, j) / norm);
                }
            } else {
                r.set(j, j, 0.0);
            }
        }

        return .{ .q = q, .r = r };
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

// Optimization algorithms
pub const opt = struct {
    /// Simple gradient descent optimizer
    /// grad_f: gradient function (takes a vector parameter, returns gradient vector)
    /// x0: initial guess
    /// learning_rate: step size
    /// max_iters: maximum iterations
    /// tolerance: convergence tolerance
    pub fn gradientDescent(
        allocator: std.mem.Allocator,
        grad_f: *const fn (std.mem.Allocator, []f64) anyerror!Vector,
        x0: []const f64,
        learning_rate: f64,
        max_iters: usize,
        tolerance: f64,
    ) !Vector {
        var x = try Vector.initWithData(allocator, x0);
        var x_new = try Vector.init(allocator, x0.len);

        var iter: usize = 0;
        while (iter < max_iters) {
            // Compute gradient
            var grad = try grad_f(allocator, x.data);
            defer grad.deinit();

            // Update: x_new = x - learning_rate * grad
            for (0..x.size()) |i| {
                const new_val = x.get(i) - learning_rate * grad.get(i);
                x_new.set(i, new_val);
            }

            // Check convergence
            var diff_norm: f64 = 0.0;
            for (0..x.size()) |i| {
                const diff = x_new.get(i) - x.get(i);
                diff_norm += diff * diff;
            }
            diff_norm = @sqrt(diff_norm);

            if (diff_norm < tolerance) {
                break;
            }

            // Swap x and x_new
            const temp = x;
            x = x_new;
            x_new = temp;

            iter += 1;
        }

        x_new.deinit();
        return x;
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

test "LU decomposition" {
    const allocator = std.testing.allocator;

    // Test matrix: [4, 3; 6, 3]
    const data = [_]f64{ 4.0, 3.0, 6.0, 3.0 };
    var a = try Matrix.initWithData(allocator, 2, 2, &data);
    defer a.deinit();

    var lu = try mat.luDecompose(allocator, a);
    defer lu.l.deinit();
    defer lu.u.deinit();

    // Expected L: [1, 0; 1.5, 1]
    try std.testing.expectEqual(@as(f64, 1.0), lu.l.get(0, 0));
    try std.testing.expectEqual(@as(f64, 0.0), lu.l.get(0, 1));
    try std.testing.expectEqual(@as(f64, 1.5), lu.l.get(1, 0));
    try std.testing.expectEqual(@as(f64, 1.0), lu.l.get(1, 1));

    // Expected U: [4, 3; 0, -1.5]
    try std.testing.expectEqual(@as(f64, 4.0), lu.u.get(0, 0));
    try std.testing.expectEqual(@as(f64, 3.0), lu.u.get(0, 1));
    try std.testing.expectEqual(@as(f64, 0.0), lu.u.get(1, 0));
    try std.testing.expectApproxEqAbs(@as(f64, -1.5), lu.u.get(1, 1), 1e-10);

    // Verify A = L * U
    var product = try mat.multiply(allocator, lu.l, lu.u);
    defer product.deinit();
    try std.testing.expectApproxEqAbs(a.get(0, 0), product.get(0, 0), 1e-10);
    try std.testing.expectApproxEqAbs(a.get(0, 1), product.get(0, 1), 1e-10);
    try std.testing.expectApproxEqAbs(a.get(1, 0), product.get(1, 0), 1e-10);
    try std.testing.expectApproxEqAbs(a.get(1, 1), product.get(1, 1), 1e-10);
}

test "QR decomposition" {
    const allocator = std.testing.allocator;

    // Test matrix: [1, 1; 1, 0; 0, 1]
    const data = [_]f64{ 1.0, 1.0, 1.0, 0.0, 0.0, 1.0 };
    var a = try Matrix.initWithData(allocator, 3, 2, &data);
    defer a.deinit();

    var qr = try mat.qrDecompose(allocator, a);
    defer qr.q.deinit();
    defer qr.r.deinit();

    // Verify A = Q * R (approximately)
    var product = try mat.multiply(allocator, qr.q, qr.r);
    defer product.deinit();

    for (0..a.rows) |i| {
        for (0..a.cols) |j| {
            try std.testing.expectApproxEqAbs(a.get(i, j), product.get(i, j), 1e-10);
        }
    }

    // Verify Q is orthogonal (Q^T * Q = I)
    var qt = try mat.transpose(allocator, qr.q);
    defer qt.deinit();
    var qqt = try mat.multiply(allocator, qt, qr.q);
    defer qqt.deinit();

    // Should be approximately identity matrix
    for (0..2) |i| {
        for (0..2) |j| {
            if (i == j) {
                try std.testing.expectApproxEqAbs(1.0, qqt.get(i, j), 1e-10);
            } else {
                try std.testing.expectApproxEqAbs(0.0, qqt.get(i, j), 1e-10);
            }
        }
    }
}

test "Gradient descent optimization" {
    const allocator = std.testing.allocator;

    // Minimize f(x,y) = (x-2)^2 + (y-3)^2, minimum at (2,3)
    const grad_f = struct {
        fn gradient(alloc: std.mem.Allocator, params: []f64) !Vector {
            const x = params[0];
            const y = params[1];
            const grad_data = [_]f64{ 2.0 * (x - 2.0), 2.0 * (y - 3.0) };
            return Vector.initWithData(alloc, &grad_data);
        }
    }.gradient;

    // Start from (0, 0), should converge to approximately (2, 3)
    const x0 = [_]f64{ 0.0, 0.0 };
    var result = try opt.gradientDescent(allocator, grad_f, &x0, 0.1, 100, 1e-6);
    defer result.deinit();

    try std.testing.expectApproxEqAbs(2.0, result.get(0), 1e-3);
    try std.testing.expectApproxEqAbs(3.0, result.get(1), 1e-3);
}

test "SIMD vector operations" {
    // Test data - use multiple of 4 for full SIMD utilization
    const a_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const b_data = [_]f64{ 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 };
    var result_data: [8]f64 = undefined;

    // Test SIMD vector addition
    simd.vecAddSIMD(&a_data, &b_data, &result_data);
    try std.testing.expectEqual(@as(f64, 9.0), result_data[0]); // 1+8
    try std.testing.expectEqual(@as(f64, 9.0), result_data[1]); // 2+7
    try std.testing.expectEqual(@as(f64, 9.0), result_data[2]); // 3+6
    try std.testing.expectEqual(@as(f64, 9.0), result_data[3]); // 4+5

    // Test SIMD dot product
    const dot_result = simd.dotSIMD(&a_data, &b_data);
    const expected_dot: f64 = 1 * 8 + 2 * 7 + 3 * 6 + 4 * 5 + 5 * 4 + 6 * 3 + 7 * 2 + 8 * 1; // 128
    try std.testing.expectEqual(expected_dot, dot_result);

    // Test SIMD scalar multiplication
    var scalar_result: [8]f64 = undefined;
    simd.scalarMulSIMD(&a_data, 3.0, &scalar_result);
    try std.testing.expectEqual(@as(f64, 3.0), scalar_result[0]); // 1*3
    try std.testing.expectEqual(@as(f64, 6.0), scalar_result[1]); // 2*3
    try std.testing.expectEqual(@as(f64, 9.0), scalar_result[2]); // 3*3
    try std.testing.expectEqual(@as(f64, 12.0), scalar_result[3]); // 4*3
}

test "Eigenvalue computation" {
    const allocator = std.testing.allocator;

    // Test matrix with known eigenvalues: [2, 1; 1, 2] has eigenvalues 3 and 1
    const data = [_]f64{ 2.0, 1.0, 1.0, 2.0 };
    var a = try Matrix.initWithData(allocator, 2, 2, &data);
    defer a.deinit();

    // Test power iteration (should find dominant eigenvalue 3)
    var result = try eigen.powerIteration(allocator, a, 100, 1e-6);
    defer result.eigenvector.deinit();

    // Should be close to 3.0 (dominant eigenvalue)
    try std.testing.expectApproxEqAbs(3.0, result.eigenvalue, 1e-4);

    // Test Jacobi method for symmetric matrix
    var jacobi_result = try eigen.jacobiEigenvalues(allocator, a, 100, 1e-6);
    defer allocator.free(jacobi_result.eigenvalues);
    defer jacobi_result.eigenvectors.deinit();

    // Should find eigenvalues 3 and 1 (in some order)
    const eigenvals = jacobi_result.eigenvalues;
    var found_3 = false;
    var found_1 = false;
    for (eigenvals) |val| {
        if (@abs(val - 3.0) < 1e-4) found_3 = true;
        if (@abs(val - 1.0) < 1e-4) found_1 = true;
    }
    try std.testing.expect(found_3);
    try std.testing.expect(found_1);
}

test "FFT operations" {
    const allocator = std.testing.allocator;

    // Test signal: sine wave
    const n = 8; // Power of 2
    var signal = try allocator.alloc(f64, n);
    defer allocator.free(signal);

    for (0..n) |i| {
        const t = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n));
        signal[i] = @sin(2.0 * std.math.pi * t); // One full cycle
    }

    // Forward FFT
    const freq_domain = try fft.forward(allocator, signal);
    defer allocator.free(freq_domain);

    // Should have non-zero magnitude at frequency bin 1
    const mag = try fft.magnitude(allocator, freq_domain);
    defer allocator.free(mag);

    // Check that we have a peak at the expected frequency
    try std.testing.expect(mag[1] > mag[0]); // Frequency bin 1 should be largest
    try std.testing.expect(mag[1] > mag[2]); // Much larger than other bins

    // Inverse FFT should reconstruct original signal
    const reconstructed = try fft.inverse(allocator, freq_domain);
    defer allocator.free(reconstructed);

    // Check reconstruction accuracy
    for (0..n) |i| {
        try std.testing.expectApproxEqAbs(signal[i], reconstructed[i], 1e-10);
    }
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

// SIMD-accelerated operations
pub const simd = struct {
    /// SIMD vector addition using Zig's vector types
    /// Only works when length is multiple of SIMD width (4 for f64)
    pub fn vecAddSIMD(a: []const f64, b: []const f64, result: []f64) void {
        const Vec4 = @Vector(4, f64);
        const len = a.len / 4 * 4; // Process in chunks of 4

        var i: usize = 0;
        while (i < len) : (i += 4) {
            const va = Vec4{ a[i], a[i + 1], a[i + 2], a[i + 3] };
            const vb = Vec4{ b[i], b[i + 1], b[i + 2], b[i + 3] };
            const vresult = va + vb;
            result[i] = vresult[0];
            result[i + 1] = vresult[1];
            result[i + 2] = vresult[2];
            result[i + 3] = vresult[3];
        }

        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            result[i] = a[i] + b[i];
        }
    }

    /// SIMD dot product using vectorized operations
    pub fn dotSIMD(a: []const f64, b: []const f64) f64 {
        const Vec4 = @Vector(4, f64);
        const len = a.len / 4 * 4;

        var sum_vec: Vec4 = @splat(0.0);
        var i: usize = 0;
        while (i < len) : (i += 4) {
            const va = Vec4{ a[i], a[i + 1], a[i + 2], a[i + 3] };
            const vb = Vec4{ b[i], b[i + 1], b[i + 2], b[i + 3] };
            sum_vec += va * vb;
        }

        var total: f64 = @reduce(.Add, sum_vec);

        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            total += a[i] * b[i];
        }

        return total;
    }

    /// SIMD scalar multiplication
    pub fn scalarMulSIMD(a: []const f64, scalar: f64, result: []f64) void {
        const Vec4 = @Vector(4, f64);
        const len = a.len / 4 * 4;
        const s_vec: Vec4 = @splat(scalar);

        var i: usize = 0;
        while (i < len) : (i += 4) {
            const va = Vec4{ a[i], a[i + 1], a[i + 2], a[i + 3] };
            const vresult = va * s_vec;
            result[i] = vresult[0];
            result[i + 1] = vresult[1];
            result[i + 2] = vresult[2];
            result[i + 3] = vresult[3];
        }

        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            result[i] = a[i] * scalar;
        }
    }
};

// Eigenvalue and eigenvector computation
pub const eigen = struct {
    /// Power iteration method for finding dominant eigenvalue and eigenvector
    /// Returns the dominant eigenvalue and corresponding eigenvector
    pub fn powerIteration(allocator: std.mem.Allocator, a: Matrix, max_iters: usize, tolerance: f64) !struct { eigenvalue: f64, eigenvector: Vector } {
        if (a.rows != a.cols) {
            return error.NotSquareMatrix;
        }

        const n = a.rows;

        // Initialize random eigenvector
        var v = try Vector.init(allocator, n);
        for (0..n) |i| {
            v.set(i, 1.0); // Start with all ones for simplicity
        }

        var eigenvalue: f64 = 0.0;
        var iter: usize = 0;

        while (iter < max_iters) {
            // Compute A * v
            var av = try Vector.init(allocator, n);
            for (0..n) |i| {
                var sum: f64 = 0.0;
                for (0..n) |j| {
                    sum += a.get(i, j) * v.get(j);
                }
                av.set(i, sum);
            }

            // Normalize Av to get new eigenvector
            var norm: f64 = 0.0;
            for (0..n) |i| {
                norm += av.get(i) * av.get(i);
            }
            norm = @sqrt(norm);

            if (norm < 1e-10) {
                av.deinit();
                return error.MatrixHasZeroEigenvalue;
            }

            var v_new = try Vector.init(allocator, n);
            for (0..n) |i| {
                v_new.set(i, av.get(i) / norm);
            }

            // Rayleigh quotient for eigenvalue estimate
            var rayleigh: f64 = 0.0;
            for (0..n) |i| {
                rayleigh += v_new.get(i) * av.get(i);
            }

            // Check convergence
            const diff: f64 = @abs(rayleigh - eigenvalue);
            eigenvalue = rayleigh;

            // Swap vectors
            const temp = v;
            v = v_new;
            temp.deinit();
            av.deinit();

            if (diff < tolerance) {
                break;
            }

            iter += 1;
        }

        return .{ .eigenvalue = eigenvalue, .eigenvector = v };
    }

    /// Jacobi eigenvalue algorithm for symmetric matrices
    /// Returns eigenvalues and eigenvectors
    pub fn jacobiEigenvalues(allocator: std.mem.Allocator, a: Matrix, max_iters: usize, tolerance: f64) !struct { eigenvalues: []f64, eigenvectors: Matrix } {
        if (a.rows != a.cols) {
            return error.NotSquareMatrix;
        }

        const n = a.rows;

        // Copy matrix for modification
        var a_copy = try Matrix.init(allocator, n, n);
        for (0..n) |i| {
            for (0..n) |j| {
                a_copy.set(i, j, a.get(i, j));
            }
        }

        // Initialize eigenvectors as identity matrix
        var eigenvectors = try Matrix.init(allocator, n, n);
        for (0..n) |i| {
            eigenvectors.set(i, i, 1.0);
        }

        var eigenvalues = try allocator.alloc(f64, n);
        @memset(eigenvalues, 0.0);

        var iter: usize = 0;
        while (iter < max_iters) {
            // Find largest off-diagonal element
            var max_val: f64 = 0.0;
            var p: usize = 0;
            var q: usize = 1;

            for (0..n) |i| {
                for (i + 1..n) |j| {
                    const val = @abs(a_copy.get(i, j));
                    if (val > max_val) {
                        max_val = val;
                        p = i;
                        q = j;
                    }
                }
            }

            // Check convergence
            if (max_val < tolerance) {
                break;
            }

            // Compute rotation parameters
            const app = a_copy.get(p, p);
            const aqq = a_copy.get(q, q);
            const apq = a_copy.get(p, q);

            const theta = 0.5 * std.math.atan2(2.0 * apq, aqq - app);
            const c = @cos(theta);
            const s = @sin(theta);

            // Update matrix A
            for (0..n) |i| {
                if (i != p and i != q) {
                    const aip = a_copy.get(i, p);
                    const aiq = a_copy.get(i, q);
                    a_copy.set(i, p, aip * c - aiq * s);
                    a_copy.set(p, i, aip * c - aiq * s);
                    a_copy.set(i, q, aiq * c + aip * s);
                    a_copy.set(q, i, aiq * c + aip * s);
                }
            }

            a_copy.set(p, p, app * c * c + aqq * s * s - 2.0 * apq * s * c);
            a_copy.set(q, q, app * s * s + aqq * c * c + 2.0 * apq * s * c);
            a_copy.set(p, q, 0.0);
            a_copy.set(q, p, 0.0);

            // Update eigenvectors
            for (0..n) |i| {
                const vip = eigenvectors.get(i, p);
                const viq = eigenvectors.get(i, q);
                eigenvectors.set(i, p, vip * c - viq * s);
                eigenvectors.set(i, q, viq * c + vip * s);
            }

            iter += 1;
        }

        // Extract eigenvalues from diagonal
        for (0..n) |i| {
            eigenvalues[i] = a_copy.get(i, i);
        }

        a_copy.deinit();

        return .{ .eigenvalues = eigenvalues, .eigenvectors = eigenvectors };
    }
};

// Fast Fourier Transform
pub const fft = struct {
    /// Complex number type for FFT
    pub const Complex = struct {
        real: f64,
        imag: f64,

        pub fn init(real: f64, imag: f64) Complex {
            return .{ .real = real, .imag = imag };
        }

        pub fn add(a: Complex, b: Complex) Complex {
            return .{ .real = a.real + b.real, .imag = a.imag + b.imag };
        }

        pub fn sub(a: Complex, b: Complex) Complex {
            return .{ .real = a.real - b.real, .imag = a.imag - b.imag };
        }

        pub fn mul(a: Complex, b: Complex) Complex {
            return .{
                .real = a.real * b.real - a.imag * b.imag,
                .imag = a.real * b.imag + a.imag * b.real,
            };
        }

        pub fn mulScalar(a: Complex, scalar: f64) Complex {
            return .{ .real = a.real * scalar, .imag = a.imag * scalar };
        }
    };

    /// Cooley-Tukey FFT algorithm for power-of-2 sizes
    /// input: real-valued signal
    /// output: complex frequency domain representation
    pub fn forward(allocator: std.mem.Allocator, input: []const f64) ![]Complex {
        const n = input.len;
        if (!std.math.isPowerOfTwo(n)) {
            return error.SizeNotPowerOfTwo;
        }

        var output = try allocator.alloc(Complex, n);

        // Initialize with input data (real part)
        for (0..n) |i| {
            output[i] = Complex.init(input[i], 0.0);
        }

        // Bit-reversal permutation
        var j: usize = 0;
        for (1..n) |i| {
            var bit = n >> 1;
            while ((j & bit) != 0) {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;

            if (i < j) {
                const temp = output[i];
                output[i] = output[j];
                output[j] = temp;
            }
        }

        // Cooley-Tukey FFT
        var length: usize = 2;
        while (length <= n) {
            const angle = -2.0 * std.math.pi / @as(f64, @floatFromInt(length));
            const wlen = Complex.init(@cos(angle), @sin(angle));

            var i: usize = 0;
            while (i < n) {
                var w = Complex.init(1.0, 0.0);

                for (0..length / 2) |k| {
                    const u = output[i + k];
                    const v = Complex.mul(output[i + k + length / 2], w);
                    output[i + k] = Complex.add(u, v);
                    output[i + k + length / 2] = Complex.sub(u, v);
                    w = Complex.mul(w, wlen);
                }

                i += length;
            }

            length <<= 1;
        }

        return output;
    }

    /// Inverse FFT
    pub fn inverse(allocator: std.mem.Allocator, input: []const Complex) ![]f64 {
        const n = input.len;
        if (!std.math.isPowerOfTwo(n)) {
            return error.SizeNotPowerOfTwo;
        }

        // Conjugate input for inverse
        var conj_input = try allocator.alloc(Complex, n);
        defer allocator.free(conj_input);

        for (0..n) |i| {
            conj_input[i] = Complex.init(input[i].real, -input[i].imag);
        }

        // Forward FFT on conjugated input
        const fft_result = try fftComplex(allocator, conj_input);
        defer allocator.free(fft_result);

        // Conjugate result and scale
        var output = try allocator.alloc(f64, n);
        const scale = 1.0 / @as(f64, @floatFromInt(n));

        for (0..n) |i| {
            output[i] = fft_result[i].real * scale;
        }

        return output;
    }

    /// FFT for complex input (helper function)
    fn fftComplex(allocator: std.mem.Allocator, input: []const Complex) ![]Complex {
        const n = input.len;
        if (!std.math.isPowerOfTwo(n)) {
            return error.SizeNotPowerOfTwo;
        }

        var output = try allocator.alloc(Complex, n);

        // Copy input
        for (0..n) |i| {
            output[i] = input[i];
        }

        // Bit-reversal permutation
        var j: usize = 0;
        for (1..n) |i| {
            var bit = n >> 1;
            while ((j & bit) != 0) {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;

            if (i < j) {
                const temp = output[i];
                output[i] = output[j];
                output[j] = temp;
            }
        }

        // Cooley-Tukey FFT
        var length: usize = 2;
        while (length <= n) {
            const angle = -2.0 * std.math.pi / @as(f64, @floatFromInt(length));
            const wlen = Complex.init(@cos(angle), @sin(angle));

            var i: usize = 0;
            while (i < n) {
                var w = Complex.init(1.0, 0.0);

                for (0..length / 2) |k| {
                    const u = output[i + k];
                    const v = Complex.mul(output[i + k + length / 2], w);
                    output[i + k] = Complex.add(u, v);
                    output[i + k + length / 2] = Complex.sub(u, v);
                    w = Complex.mul(w, wlen);
                }

                i += length;
            }

            length <<= 1;
        }

        return output;
    }

    /// Compute magnitude spectrum
    pub fn magnitude(allocator: std.mem.Allocator, freq_domain: []const Complex) ![]f64 {
        var mag = try allocator.alloc(f64, freq_domain.len);
        for (0..freq_domain.len) |i| {
            const c = freq_domain[i];
            mag[i] = @sqrt(c.real * c.real + c.imag * c.imag);
        }
        return mag;
    }
};
