package com.hopeproject.influencer.sem.weka;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;

/** Cross-software validation against lavaan's observed-information ML solution. */
class LavaanReferenceValidationTest {

    private static final Path REFERENCE = Path.of("validation", "results", "lavaan-reference.csv");

    static Stream<ValidationCase> referenceCases() {
        return Stream.of(
                new ValidationCase(
                        "holzinger-swineford-1939",
                        "validation/data/holzinger-swineford-1939.arff",
                        "visual=x1,x2,x3;textual=x4,x5,x6;speed=x7,x8,x9"),
                new ValidationCase(
                        "political-democracy",
                        "validation/data/political-democracy.arff",
                        "ind60=x1,x2,x3;dem60=y1,y2,y3,y4")
        );
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("referenceCases")
    void agreesWithLavaanWithinDocumentedTolerances(ValidationCase validationCase) throws Exception {
        Instances data = DataSource.read(validationCase.arff);
        data.setClassIndex(data.numAttributes() - 1);
        CovarianceSemModel model = new CovarianceSemEstimator().fit(
                data,
                CompositeSpecification.parse(validationCase.specification),
                1.0e-10,
                1_500,
                1.0e-8,
                true);
        Reference reference = readReference(validationCase.id);

        assertThat(model.converged()).isTrue();
        assertClose(model.chiSquare(), reference.fit.get("chisq"), 0.02, "chi-square");
        assertClose(model.degreesOfFreedom(), reference.fit.get("df"), 0.0, "degrees of freedom");
        assertClose(model.chiSquarePValue(), reference.fit.get("pvalue"), 0.001, "chi-square p-value");
        assertClose(model.cfi(), reference.fit.get("cfi"), 0.001, "CFI");
        assertClose(model.tli(), reference.fit.get("tli"), 0.001, "TLI");
        assertClose(model.rmsea(), reference.fit.get("rmsea"), 0.001, "RMSEA");
        assertClose(model.srmr(), reference.fit.get("srmr"), 0.001, "SRMR");
        assertClose(model.aic(), reference.fit.get("aic"), 0.02, "AIC");
        assertClose(model.bic(), reference.fit.get("bic"), 0.02, "BIC");

        assertThat(model.parameterEstimates()).isNotEmpty();
        for (SemParameterEstimate actual : model.parameterEstimates()) {
            ParameterReference expected = reference.parameter(actual.label());
            assertThat(expected)
                    .as("lavaan reference for %s", actual.label())
                    .isNotNull();
            assertClose(actual.estimate(), expected.estimate, 0.005, actual.label() + " estimate");
            assertClose(actual.standardizedEstimate(), expected.standardized, 0.005,
                    actual.label() + " std.all");
            if (actual.free()) {
                assertThat(actual.standardError()).as(actual.label() + " standard error").isFinite();
                assertThat(actual.zStatistic()).as(actual.label() + " z statistic").isFinite();
                assertThat(actual.pValue()).as(actual.label() + " p-value").isBetween(0.0, 1.0);
                assertClose(actual.standardError(), expected.standardError, 0.005,
                        actual.label() + " standard error");
                assertClose(actual.zStatistic(), expected.z, 0.05, actual.label() + " z statistic");
                assertClose(actual.pValue(), expected.pValue, 0.005, actual.label() + " p-value");
            } else {
                assertThat(actual.standardError()).isZero();
                assertThat(actual.zStatistic()).isNaN();
                assertThat(actual.pValue()).isNaN();
            }
        }
    }

    private Reference readReference(String dataset) throws Exception {
        Map<String, Double> fit = new HashMap<>();
        Map<String, ParameterReference> parameters = new HashMap<>();
        List<String> lines = Files.readAllLines(REFERENCE);
        for (int index = 1; index < lines.size(); index++) {
            String[] cells = csvCells(lines.get(index));
            if (!cells[0].equals(dataset)) continue;
            if (cells[1].equals("fit")) {
                fit.put(cells[2], parse(cells[6]));
            } else {
                ParameterReference parameter = new ParameterReference(
                        parse(cells[6]), parse(cells[7]), parse(cells[8]),
                        parse(cells[9]), parse(cells[10]));
                parameters.put(cells[2], parameter);
                if (cells[4].equals("~~") && !cells[3].equals(cells[5])) {
                    parameters.put(cells[5] + "~~" + cells[3], parameter);
                }
            }
        }
        return new Reference(fit, parameters);
    }

    private String[] csvCells(String line) {
        String[] raw = line.split(",", -1);
        List<String> result = new ArrayList<>(raw.length);
        for (String cell : raw) {
            result.add(cell.length() >= 2 && cell.startsWith("\"") && cell.endsWith("\"")
                    ? cell.substring(1, cell.length() - 1).replace("\"\"", "\"")
                    : cell);
        }
        return result.toArray(String[]::new);
    }

    private double parse(String value) {
        return value.isBlank() ? Double.NaN : Double.parseDouble(value);
    }

    private void assertClose(double actual, double expected, double tolerance, String description) {
        assertThat(actual).as(description).isCloseTo(expected,
                org.assertj.core.data.Offset.offset(tolerance));
    }

    private record ValidationCase(String id, String arff, String specification) {
        @Override
        public String toString() {
            return id;
        }
    }

    private record ParameterReference(
            double estimate,
            double standardError,
            double z,
            double pValue,
            double standardized) {
    }

    private record Reference(Map<String, Double> fit, Map<String, ParameterReference> parameters) {
        private ParameterReference parameter(String label) {
            return parameters.get(label);
        }
    }
}
