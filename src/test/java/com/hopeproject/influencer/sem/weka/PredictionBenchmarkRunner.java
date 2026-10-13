package com.hopeproject.influencer.sem.weka;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.PLSClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.supervised.attribute.PLSFilter;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/** Generates deterministic prediction benchmarks on the public SEM fixtures. */
public final class PredictionBenchmarkRunner {

    private static final int[] SEEDS = {1, 17, 42};
    private static final int FOLDS = 10;

    private PredictionBenchmarkRunner() {
    }

    /**
     * Writes the benchmark CSV. The optional first argument overrides the
     * default {@code validation/results/prediction-benchmarks.csv} path.
     *
     * @param args optional output path
     * @throws Exception if a dataset or classifier cannot be evaluated
     */
    public static void main(String[] args) throws Exception {
        Path output = args.length == 0
                ? Path.of("validation", "results", "prediction-benchmarks.csv")
                : Path.of(args[0]);
        Files.createDirectories(output.toAbsolutePath().getParent());

        List<BenchmarkCase> datasets = List.of(
                new BenchmarkCase(
                        "holzinger-swineford-1939",
                        "validation/data/holzinger-swineford-1939.arff",
                        "visual=x1,x2,x3;textual=x4,x5,x6;speed=x7,x8,x9"),
                new BenchmarkCase(
                        "political-democracy",
                        "validation/data/political-democracy.arff",
                        "ind60=x1,x2,x3;dem60=y1,y2,y3,y4")
        );

        List<String> lines = new ArrayList<>();
        lines.add("dataset,model,repetitions,folds,rmse,mae,correlation");
        for (BenchmarkCase benchmark : datasets) {
            Instances data = DataSource.read(benchmark.arff);
            data.setClassIndex(data.numAttributes() - 1);
            for (ModelFactory model : models(benchmark.specification, data.numAttributes() - 1)) {
                Metrics metrics = evaluate(data, model);
                lines.add(String.format(Locale.ROOT, "%s,%s,%d,%d,%.9f,%.9f,%.9f",
                        benchmark.id, model.name, SEEDS.length, FOLDS,
                        metrics.rmse, metrics.mae, metrics.correlation));
            }
        }
        Files.write(output, lines, StandardCharsets.UTF_8);
        lines.forEach(System.out::println);
    }

    private static List<ModelFactory> models(String specification, int predictors) {
        return List.of(
                new ModelFactory("CovarianceSEMRegressor", () -> {
                    CovarianceSEMRegressor model = new CovarianceSEMRegressor();
                    model.setConstructs(specification);
                    model.setRegularization(1.0e-6);
                    model.setMaximumIterations(600);
                    model.setTolerance(1.0e-6);
                    model.setComputeInference(false);
                    return model;
                }),
                new ModelFactory("LinearRegression", LinearRegression::new),
                new ModelFactory("RandomForest", () -> {
                    RandomForest model = new RandomForest();
                    model.setNumIterations(300);
                    model.setSeed(1);
                    return model;
                }),
                new ModelFactory("partialLeastSquares-SIMPLS", () -> {
                    PLSFilter filter = new PLSFilter();
                    filter.setNumComponents(Math.min(5, predictors));
                    filter.setAlgorithm(new SelectedTag(PLSFilter.ALGORITHM_SIMPLS,
                            PLSFilter.TAGS_ALGORITHM));
                    filter.setPreprocessing(new SelectedTag(PLSFilter.PREPROCESSING_STANDARDIZE,
                            PLSFilter.TAGS_PREPROCESSING));
                    PLSClassifier model = new PLSClassifier();
                    model.setFilter(filter);
                    return model;
                })
        );
    }

    private static Metrics evaluate(Instances data, ModelFactory model) throws Exception {
        double rmse = 0.0;
        double mae = 0.0;
        double correlation = 0.0;
        for (int seed : SEEDS) {
            Evaluation evaluation = new Evaluation(data);
            evaluation.crossValidateModel(model.factory.create(), data, FOLDS, new Random(seed));
            rmse += evaluation.rootMeanSquaredError();
            mae += evaluation.meanAbsoluteError();
            correlation += evaluation.correlationCoefficient();
        }
        return new Metrics(rmse / SEEDS.length, mae / SEEDS.length, correlation / SEEDS.length);
    }

    @FunctionalInterface
    private interface ThrowingFactory {
        Classifier create() throws Exception;
    }

    private record ModelFactory(String name, ThrowingFactory factory) {
    }

    private record BenchmarkCase(String id, String arff, String specification) {
    }

    private record Metrics(double rmse, double mae, double correlation) {
    }
}
