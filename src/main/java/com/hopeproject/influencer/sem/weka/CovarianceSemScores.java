package com.hopeproject.influencer.sem.weka;

import java.io.Serial;
import java.io.Serializable;

/** Factor scores and SEM prediction produced for one Weka instance. */
public record CovarianceSemScores(
        String[] constructNames,
        double[] factorScores,
        double[] standardizedFactorScores,
        double standardizedPrediction,
        double prediction
) implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    public CovarianceSemScores {
        constructNames = constructNames.clone();
        factorScores = factorScores.clone();
        standardizedFactorScores = standardizedFactorScores.clone();
    }
}
