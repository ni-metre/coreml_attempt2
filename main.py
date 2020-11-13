import tensorflow as tf
import tensorflow_hub as tf_hub
import numpy as np
import coremltools as ct




def main():
  model = tf.keras.Sequential([
          tf.keras.layers.InputLayer(input_shape=(192, 192, 3)),
          tf_hub.KerasLayer(
            "https://tfhub.dev/google/imagenet/mobilenet_v2_050_192/classification/4"
          )
  ])
  
  model.build([1, 192, 192, 3])  # Batch input shape.
  # random input data to check that predict works
  x = np.random.rand(1, 192, 192, 3)
  tf_out = model.predict([x])
  # convert to Core ML and check predictions
  mlmodel = ct.convert(model)
  coreml_out_dict = mlmodel.predict({"image":x})
  coreml_out = list(coreml_out_dict.values())[0]
  np.testing.assert_allclose(tf_out, coreml_out, rtol=1e-2, atol=1e-1)
  
  # convert to an image input Core ML model
  # mobilenet model expects images to be normalized in the interval [-1,1]
  # hence bias of -1 and scale of 1/127
  mlmodel = ct.convert(model,
                      inputs=[ct.ImageType(bias=[-1,-1,-1], scale=1/127)])
  
  mlmodel.save("mobilenet.mlmodel")
  print("this worked")
  
if __name__ == '__main__':
	main()
  
