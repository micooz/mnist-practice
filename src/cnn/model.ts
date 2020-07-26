import * as tf from '@tensorflow/tfjs-node';

export default function createModel() {
  const model = tf.sequential();

  model.add(
    tf.layers.conv2d({
      inputShape: [28, 28, 1],
      filters: 8,
      kernelSize: 3,
    }),
  );

  model.add(
    tf.layers.maxPooling2d({
      poolSize: 2,
    }),
  );

  model.add(tf.layers.flatten());

  model.add(
    tf.layers.dense({
      units: 10,
      activation: 'softmax',
    }),
  );

  return model;
}
