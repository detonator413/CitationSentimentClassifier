/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.chaoticity.citationsentiment;

import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;

import java.util.Arrays;

/**
 * Code and data for citation sentiment classification reported in http://www.aclweb.org/anthology/P11-3015
 * The file test.arff contains only the test set with dependency triplets generated with Stanford CoreNLP
 * Full corpus available at http://www.cl.cam.ac.uk/~aa496/citation-sentiment-corpus
 *
 * @author Awais Athar
 */
public class CitationSentimentTester {


    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        FilteredClassifier svm = (FilteredClassifier) SerializationHelper.read("/tmp/citmodel.dat");
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("example.arff");
        Instances data = source.getDataSet();

        data.setClassIndex(0);
        data.deleteAttributeAt(1);
        data.deleteAttributeAt(2);

        for (int i = 0; i < data.numInstances(); i++) {
            double[] res = svm.distributionForInstance(data.instance(i));
            System.out.println(Arrays.toString(res));
            System.out.println(svm.classifyInstance(data.instance(i)));
        }


    }



}
