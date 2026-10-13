package com.hopeproject.influencer.sem.weka;

import java.util.Arrays;
import java.util.function.ToDoubleFunction;

/** Dependency-free BFGS optimizer with numerical derivatives and an Armijo line search. */
final class SemBfgsOptimizer {

    record Result(double[] parameters, double value, double[][] inverseHessian,
                  int iterations, boolean converged, double gradientNorm) {
    }

    Result optimize(ToDoubleFunction<double[]> function, double[] initial,
                    int maximumIterations, double tolerance) {
        double[] x = initial.clone();
        int size = x.length;
        double[][] inverseHessian = SemMatrixOps.identity(size);
        double value = function.applyAsDouble(x);
        double[] gradient = gradient(function, x);
        boolean converged = false;
        int iteration = 0;

        for (; iteration < maximumIterations; iteration++) {
            if (SemMatrixOps.normInfinity(gradient) < tolerance) {
                converged = true;
                break;
            }
            double[] direction = negate(SemMatrixOps.multiply(inverseHessian, gradient));
            double directionalDerivative = SemMatrixOps.dot(gradient, direction);
            if (!(directionalDerivative < 0) || !Double.isFinite(directionalDerivative)) {
                inverseHessian = SemMatrixOps.identity(size);
                direction = negate(gradient);
                directionalDerivative = -SemMatrixOps.dot(gradient, gradient);
            }

            double step = 1.0;
            double[] candidate = null;
            double candidateValue = Double.POSITIVE_INFINITY;
            for (int search = 0; search < 35; search++) {
                candidate = add(x, direction, step);
                candidateValue = function.applyAsDouble(candidate);
                if (Double.isFinite(candidateValue)
                        && candidateValue <= value + 1.0e-4 * step * directionalDerivative) break;
                step *= 0.5;
            }
            if (candidate == null || !Double.isFinite(candidateValue) || step < 1.0e-10) break;

            double[] nextGradient = gradient(function, candidate);
            double[] s = subtract(candidate, x);
            double[] y = subtract(nextGradient, gradient);
            double ys = SemMatrixOps.dot(y, s);
            if (ys > 1.0e-12 && Double.isFinite(ys)) {
                inverseHessian = bfgsInverseUpdate(inverseHessian, s, y, ys);
            } else {
                inverseHessian = SemMatrixOps.identity(size);
            }
            double improvement = Math.abs(value - candidateValue);
            x = candidate;
            value = candidateValue;
            gradient = nextGradient;
            if (improvement < tolerance * (1.0 + Math.abs(value))
                    && SemMatrixOps.normInfinity(gradient) < Math.sqrt(tolerance)) {
                converged = true;
                iteration++;
                break;
            }
        }
        return new Result(x, value, inverseHessian, iteration, converged,
                SemMatrixOps.normInfinity(gradient));
    }

    private double[] gradient(ToDoubleFunction<double[]> function, double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            double step = 1.0e-5 * (1.0 + Math.abs(x[i]));
            double[] plus = x.clone();
            double[] minus = x.clone();
            plus[i] += step;
            minus[i] -= step;
            double plusValue = function.applyAsDouble(plus);
            double minusValue = function.applyAsDouble(minus);
            result[i] = (plusValue - minusValue) / (2.0 * step);
            if (!Double.isFinite(result[i])) result[i] = 0.0;
        }
        return result;
    }

    private double[][] bfgsInverseUpdate(double[][] h, double[] s, double[] y, double ys) {
        int size = s.length;
        double rho = 1.0 / ys;
        double[][] left = SemMatrixOps.identity(size);
        double[][] right = SemMatrixOps.identity(size);
        for (int row = 0; row < size; row++) {
            for (int column = 0; column < size; column++) {
                left[row][column] -= rho * s[row] * y[column];
                right[row][column] -= rho * y[row] * s[column];
            }
        }
        double[][] updated = SemMatrixOps.multiply(SemMatrixOps.multiply(left, h), right);
        for (int row = 0; row < size; row++) {
            for (int column = 0; column < size; column++) {
                updated[row][column] += rho * s[row] * s[column];
            }
        }
        return updated;
    }

    private double[] negate(double[] values) {
        return Arrays.stream(values).map(value -> -value).toArray();
    }

    private double[] add(double[] left, double[] right, double multiplier) {
        double[] result = new double[left.length];
        for (int i = 0; i < left.length; i++) result[i] = left[i] + multiplier * right[i];
        return result;
    }

    private double[] subtract(double[] left, double[] right) {
        double[] result = new double[left.length];
        for (int i = 0; i < left.length; i++) result[i] = left[i] - right[i];
        return result;
    }
}
