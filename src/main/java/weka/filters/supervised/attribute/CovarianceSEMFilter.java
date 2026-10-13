package weka.filters.supervised.attribute;

/** Weka package-discovery entry point. */
public class CovarianceSEMFilter
        extends com.hopeproject.influencer.sem.weka.CovarianceSEMFilter {

    private static final long serialVersionUID = 1L;

    public static void main(String[] args) {
        runFilter(new CovarianceSEMFilter(), args);
    }
}
