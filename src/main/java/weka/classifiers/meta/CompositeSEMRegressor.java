package weka.classifiers.meta;

/**
 * Weka-native discovery alias for the generic Composite SEM meta-regressor.
 */
public class CompositeSEMRegressor
        extends com.hopeproject.influencer.sem.weka.CompositeSEMRegressor {

    private static final long serialVersionUID = 1L;

    public static void main(String[] args) {
        runClassifier(new CompositeSEMRegressor(), args);
    }
}
