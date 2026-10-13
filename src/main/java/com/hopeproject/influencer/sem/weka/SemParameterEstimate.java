package com.hopeproject.influencer.sem.weka;

import java.io.Serial;
import java.io.Serializable;
import java.util.Objects;

/**
 * One covariance-SEM parameter with Wald inference and a fully standardized
 * estimate.
 *
 * <p>The {@code operator} follows the conventional SEM notation used by
 * lavaan: {@code =~} for factor loadings, {@code ~} for structural paths and
 * {@code ~~} for variances or covariances. Estimates and standard errors are
 * returned on the original observed-variable scale. {@code standardizedEstimate}
 * is the completely standardized ({@code std.all}) solution.</p>
 *
 * @param lhs left-hand variable or construct
 * @param operator SEM operator ({@code =~}, {@code ~}, or {@code ~~})
 * @param rhs right-hand variable or construct
 * @param estimate unstandardized estimate on the original data scale
 * @param standardError delta-method standard error; {@link Double#NaN} for a fixed parameter
 * @param zStatistic Wald z statistic; {@link Double#NaN} for a fixed parameter
 * @param pValue two-sided normal-theory p-value; {@link Double#NaN} for a fixed parameter
 * @param standardizedEstimate completely standardized estimate
 * @param free whether the parameter was freely estimated
 */
public record SemParameterEstimate(
        String lhs,
        String operator,
        String rhs,
        double estimate,
        double standardError,
        double zStatistic,
        double pValue,
        double standardizedEstimate,
        boolean free
) implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    /** Validates the textual parts of the parameter description. */
    public SemParameterEstimate {
        Objects.requireNonNull(lhs, "lhs");
        Objects.requireNonNull(operator, "operator");
        Objects.requireNonNull(rhs, "rhs");
    }

    /** Returns the conventional compact SEM label, for example {@code visual=~x1}. */
    public String label() {
        return lhs + operator + rhs;
    }
}
