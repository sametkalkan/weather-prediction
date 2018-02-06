package com.kalkan.weatherprediction;

/**
 * Created by Samet KALKAN
 */

public class Classification {

    private float probability;
    private int classIndex;


    public void process_output(float[] output){
        setClassIndex(argmax(output));
        setProbability(output[classIndex]);
    }

    private int argmax(float[] array){
        int max = 0;
        for(int i=0;i<array.length;i++)
            if(array[i]>array[max])
                max = i;
        return max;
    }

    public float getProbability() {
        return probability;
    }
    public void setProbability(float probability) {
        this.probability = probability;
    }
    public int getClassIndex() {
        return classIndex;
    }
    public void setClassIndex(int classIndex) {
        this.classIndex = classIndex;
    }
}
