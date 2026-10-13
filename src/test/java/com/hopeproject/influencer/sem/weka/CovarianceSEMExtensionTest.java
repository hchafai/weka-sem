package com.hopeproject.influencer.sem.weka;

import org.junit.jupiter.api.Test;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.filters.Filter;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;

class CovarianceSEMExtensionTest {

    private static final String SPECIFICATION =
            "Activation=engagement,comments,sentiment;Reach=followers,views,frequency";

    @Test
    void estimatesCovarianceModelAndFitIndices() {
        Instances data = dataset(320, 23);
        CovarianceSemModel model = new CovarianceSemEstimator().fit(
                data, CompositeSpecification.parse(SPECIFICATION), 1.0e-6, 400, 1.0e-6);

        assertThat(model.converged()).isTrue();
        assertThat(model.degreesOfFreedom()).isPositive();
        assertThat(model.chiSquare()).isFinite().isGreaterThanOrEqualTo(0);
        assertThat(model.cfi()).isBetween(0.0, 1.0);
        assertThat(model.tli()).isBetween(0.0, 1.0);
        assertThat(model.rmsea()).isFinite().isGreaterThanOrEqualTo(0);
        assertThat(model.srmr()).isFinite().isLessThan(0.15);
        assertThat(model.rSquared()).isGreaterThan(0.35);
        assertThat(model.loadings()).hasDimensions(2, 3);
        assertThat(model.latentCovariance()).hasDimensions(2, 2);
        assertThat(model.score(data.instance(0)).prediction()).isFinite();
    }

    @Test
    void worksAsWekaRegressorAndSerializableModel() throws Exception {
        Instances data = dataset(220, 71);
        CovarianceSEMRegressor regressor = new CovarianceSEMRegressor();
        regressor.setConstructs(SPECIFICATION);
        regressor.setMaximumIterations(350);

        Evaluation evaluation = new Evaluation(data);
        evaluation.crossValidateModel(regressor, data, 3, new Random(5));
        assertThat(evaluation.rootMeanSquaredError()).isFinite().isLessThan(2.0);

        regressor.buildClassifier(data);
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        SerializationHelper.write(output, regressor);
        CovarianceSEMRegressor restored = (CovarianceSEMRegressor) SerializationHelper.read(
                new ByteArrayInputStream(output.toByteArray()));
        assertThat(restored.classifyInstance(data.instance(3))).isFinite();
    }

    @Test
    void supervisedFilterReusesFittedSemForScoring() throws Exception {
        Instances data = dataset(200, 101);
        data.instance(4).setMissing(data.attribute("comments"));
        CovarianceSEMFilter filter = new CovarianceSEMFilter();
        filter.setConstructs(SPECIFICATION);
        filter.setInputFormat(data);

        Instances transformed = Filter.useFilter(data, filter);
        assertThat(transformed.attribute("sem_ml_activation")).isNotNull();
        assertThat(transformed.attribute("sem_ml_reach")).isNotNull();
        assertThat(transformed.attribute("sem_ml_prediction")).isNotNull();
        assertThat(filter.getModel().trainingRows()).isEqualTo(200);

        Instances scoring = new Instances(data, 0);
        scoring.add(data.instance(0));
        scoring.instance(0).setClassMissing();
        Instances scored = Filter.useFilter(scoring, filter);
        assertThat(scored).hasSize(1);
        assertThat(scored.instance(0).value(scored.attribute("sem_ml_prediction"))).isFinite();
        assertThat(filter.getModel().trainingRows()).isEqualTo(200);
    }

    private Instances dataset(int size, long seed) {
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("engagement"));
        attributes.add(new Attribute("comments"));
        attributes.add(new Attribute("sentiment"));
        attributes.add(new Attribute("followers"));
        attributes.add(new Attribute("views"));
        attributes.add(new Attribute("frequency"));
        attributes.add(new Attribute("net_value"));
        Instances data = new Instances("covariance-sem-synthetic", attributes, size);
        data.setClassIndex(data.numAttributes() - 1);

        Random random = new Random(seed);
        for (int index = 0; index < size; index++) {
            double activation = random.nextGaussian();
            double reach = 0.35 * activation + Math.sqrt(1 - 0.35 * 0.35) * random.nextGaussian();
            double[] values = {
                    activation + 0.35 * random.nextGaussian(),
                    0.85 * activation + 0.40 * random.nextGaussian(),
                    0.70 * activation + 0.45 * random.nextGaussian(),
                    reach + 0.35 * random.nextGaussian(),
                    0.90 * reach + 0.40 * random.nextGaussian(),
                    0.75 * reach + 0.45 * random.nextGaussian(),
                    0.65 * activation + 0.50 * reach + 0.45 * random.nextGaussian()
            };
            data.add(new DenseInstance(1.0, values));
        }
        return data;
    }
}
