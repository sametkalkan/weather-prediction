package com.kalkan.weatherprediction;

/**
 * Created by Samet KALKAN
 */

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;


public class MainActivity extends AppCompatActivity {

    //class number maps to a string for example, 0->"Cloudy"
    private HashMap<Integer, String> classes = new HashMap<>();

    private static final int CROP_SIZE = 50;

    private Button camera;
    private ImageView imageView;
    private TensorFlowClassifier classifier;
    private TextView predictText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initialize();

        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(intent, 0);

            }
        });

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        Bitmap imageBitmap = (Bitmap)data.getExtras().get("data");
        //Bitmap rotated = rotateImage(imageBitmap, 90);


        imageView.setImageBitmap(imageBitmap); // sets taken image from camera

        predictImage(imageBitmap); // predicts class of taken image

        String txt = String.format("Prediction: %s \nProbability: %%%.2f ",
                                    classes.get(classifier.cls.getClassIndex()),
                                    classifier.cls.getProbability()*100);
        predictText.setText(txt);
    }

    private void predictImage(Bitmap image){
        Bitmap resized = resizeImageKeepRatio(image, CROP_SIZE);
        Bitmap cropped = cropImage(resized, CROP_SIZE, CROP_SIZE);
        float[] pixels = getPixels(cropped);

        classifier.recognize(pixels);
    }

    /**
     * rotates image
     * it is necessary because photo is showed as rotated.
     * so we must rotate it.
     * @param bitmap
     * @param degree
     * @return
     */
    private Bitmap rotateImage(Bitmap bitmap, int degree){
        Matrix matrix = new Matrix();
        matrix.postRotate(degree);
        Bitmap rotated = Bitmap.createBitmap(bitmap, 0, 0,
                                        bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        return rotated;
    }

    /**
     * returns RGB values of each pixel of given bitmap image
     * @param bitmap image to be converted
     * @return pixels of an image
     */
    private float[] getPixels(Bitmap bitmap){
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        int[] intValues = new int[width * height];

        float[] floatValues = new float[width*height*3];

        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        // getting RGB values and normalization
        for (int i = 0; i < intValues.length; ++i) {
            int val = intValues[i];
            floatValues[i * 3 + 0] = (((val >> 16) & 0xFF))/255f;
            floatValues[i * 3 + 1] = ((val >> 8) & 0xFF)/255f;
            floatValues[i * 3 + 2] = (val & 0xFF)/255f;
        }

        return floatValues;
    }

    /**
     * reads an image from given path
     * this is necessary for testing images in the assets
     * it is doing nothing for this app because image comes from camera
     * @param path path of image
     * @return bitmap image
     */
    private Bitmap readImage(String path){

        InputStream istr;
        try{
            istr = getAssets().open(path);

        }catch (Exception e){
            return null;
        }

        return BitmapFactory.decodeStream(istr);
    }

    /**
     *  This is pre-processing for image.
     * resizes and crops an image for tensorflow processing.
     * @param bitmap image
     * @param crop_size crop size is width and height at the end.
     * @return resized and cropped image
     */
    private Bitmap resizeImageKeepRatio(Bitmap bitmap, int crop_size){
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        int newWidth, newHeight;

        if(height>width) {
            newWidth = crop_size;
            newHeight = (int)(crop_size * height)/width;
        }
        else{
            newWidth = (crop_size * width)/height;
            newHeight = (int)crop_size;
        }

        Bitmap resized = Bitmap.createScaledBitmap(bitmap, newWidth, newHeight,true);

        return resized;
    }

    private Bitmap cropImage(Bitmap bitmap, int width, int height){
        return Bitmap.createBitmap(bitmap, 0,0, width, height);
    }

    private void loadModel() {
        try {
            classifier = new TensorFlowClassifier(getAssets(),
                        "frozen_mymodel.pb", CROP_SIZE,
                        "input", "output");
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    private void initialize(){
        camera = findViewById(R.id.camera);
        imageView = findViewById(R.id.imageView);
        predictText = findViewById(R.id.predictText);

        initMap(); // initializing for mapping
        loadModel(); // loads tensorflow model
    }

    private void initMap(){
        classes.put(0,"Cloudy");
        classes.put(1,"Sunny");
        classes.put(2,"Rainy");
        classes.put(3,"Snowy");
        classes.put(4,"Foggy");
    }

}


















