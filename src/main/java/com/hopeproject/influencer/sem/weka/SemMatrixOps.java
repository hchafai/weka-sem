package com.hopeproject.influencer.sem.weka;

/** Small dependency-free matrix helpers used by the covariance SEM engine. */
final class SemMatrixOps {

    private static final double EPSILON = 1.0e-12;

    private SemMatrixOps() {
    }

    static double[][] identity(int size) {
        double[][] result = new double[size][size];
        for (int i = 0; i < size; i++) result[i][i] = 1.0;
        return result;
    }

    static double[][] transpose(double[][] matrix) {
        double[][] result = new double[matrix[0].length][matrix.length];
        for (int row = 0; row < matrix.length; row++) {
            for (int column = 0; column < matrix[row].length; column++) {
                result[column][row] = matrix[row][column];
            }
        }
        return result;
    }

    static double[][] multiply(double[][] left, double[][] right) {
        double[][] result = new double[left.length][right[0].length];
        for (int row = 0; row < left.length; row++) {
            for (int inner = 0; inner < right.length; inner++) {
                double value = left[row][inner];
                for (int column = 0; column < right[inner].length; column++) {
                    result[row][column] += value * right[inner][column];
                }
            }
        }
        return result;
    }

    static double[] multiply(double[][] matrix, double[] vector) {
        double[] result = new double[matrix.length];
        for (int row = 0; row < matrix.length; row++) {
            for (int column = 0; column < vector.length; column++) {
                result[row] += matrix[row][column] * vector[column];
            }
        }
        return result;
    }

    static double[][] inverse(double[][] source) {
        int size = source.length;
        double[][] augmented = new double[size][size * 2];
        for (int row = 0; row < size; row++) {
            if (source[row].length != size) throw new IllegalArgumentException("Matrix must be square");
            System.arraycopy(source[row], 0, augmented[row], 0, size);
            augmented[row][size + row] = 1.0;
        }
        for (int pivot = 0; pivot < size; pivot++) {
            int best = pivot;
            for (int row = pivot + 1; row < size; row++) {
                if (Math.abs(augmented[row][pivot]) > Math.abs(augmented[best][pivot])) best = row;
            }
            if (Math.abs(augmented[best][pivot]) < EPSILON) {
                throw new IllegalArgumentException("Matrix is singular");
            }
            double[] swap = augmented[pivot];
            augmented[pivot] = augmented[best];
            augmented[best] = swap;
            double divisor = augmented[pivot][pivot];
            for (int column = 0; column < size * 2; column++) augmented[pivot][column] /= divisor;
            for (int row = 0; row < size; row++) {
                if (row == pivot) continue;
                double factor = augmented[row][pivot];
                for (int column = 0; column < size * 2; column++) {
                    augmented[row][column] -= factor * augmented[pivot][column];
                }
            }
        }
        double[][] result = new double[size][size];
        for (int row = 0; row < size; row++) {
            System.arraycopy(augmented[row], size, result[row], 0, size);
        }
        return result;
    }

    static double logDeterminantPositiveDefinite(double[][] matrix) {
        double[][] lower = cholesky(matrix);
        double result = 0;
        for (int i = 0; i < lower.length; i++) result += 2.0 * Math.log(lower[i][i]);
        return result;
    }

    static double[][] cholesky(double[][] matrix) {
        int size = matrix.length;
        double[][] lower = new double[size][size];
        for (int row = 0; row < size; row++) {
            for (int column = 0; column <= row; column++) {
                double value = matrix[row][column];
                for (int k = 0; k < column; k++) value -= lower[row][k] * lower[column][k];
                if (row == column) {
                    if (!(value > EPSILON) || !Double.isFinite(value)) {
                        throw new IllegalArgumentException("Matrix is not positive definite");
                    }
                    lower[row][column] = Math.sqrt(value);
                } else {
                    lower[row][column] = value / lower[column][column];
                }
            }
        }
        return lower;
    }

    static double traceProduct(double[][] left, double[][] right) {
        double result = 0;
        for (int row = 0; row < left.length; row++) {
            for (int column = 0; column < left.length; column++) {
                result += left[row][column] * right[column][row];
            }
        }
        return result;
    }

    static double dot(double[] left, double[] right) {
        double result = 0;
        for (int i = 0; i < left.length; i++) result += left[i] * right[i];
        return result;
    }

    static double normInfinity(double[] values) {
        double result = 0;
        for (double value : values) result = Math.max(result, Math.abs(value));
        return result;
    }

    static double[][] copy(double[][] matrix) {
        double[][] result = new double[matrix.length][];
        for (int i = 0; i < matrix.length; i++) result[i] = matrix[i].clone();
        return result;
    }
}
