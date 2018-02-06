package com.kalkan.weatherprediction;

/**
 * Created by Samet KALKAN
 */

import android.content.res.AssetManager;
import java.io.IOException;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;


public class TensorFlowClassifier {

    Classification cls = new Classification();

    private TensorFlowInferenceInterface tf;

    private String inputName;
    private String outputName;
    private int inputSize;

    private float[] output;
    private String[] outputNames;


   public TensorFlowClassifier(AssetManager assetManager, String modelPath, int inputSize,
                               String inputName, String outputName) throws IOException {
        int numClasses = 5;

        this.inputName = inputName;
        this.outputName = outputName;

        this.inputSize = inputSize;

        this.outputNames = new String[] { outputName };
        this.outputName = outputName;

        this.output = new float[numClasses];

        this.tf = new TensorFlowInferenceInterface(assetManager, modelPath);

   }

    public void recognize(final float[] pixels) {

        tf.feed(inputName, pixels, 1, inputSize, inputSize, 3);

        tf.feed("keep_prob", new float[] { 1 });  // for dropout part

        tf.run(outputNames);  //get the possible outputs, we have only one output

        tf.fetch(outputName, output);  //get the output

        cls.process_output(output);

    }


}
