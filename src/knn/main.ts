import * as tf from '@tensorflow/tfjs-node';
import lodash from 'lodash';
import { getTrainData, getTestData } from '../dataset';

const [testImages, testLabels] = getTestData(10);
const [trainImages, trainLabels] = getTrainData(2000);

// trainLabels.argMax(1).print();

const K = 5;

let acc = 0;

let labels: number[] = [];
let predictions: number[] = [];

for (let i = 0; i < testImages.shape[0]; i++) {
  const testImage = testImages.slice(i, 1).flatten();
  const testLabel = testLabels.slice(i, 1).flatten();
  const result = [];

  for (let j = 0; j < trainImages.shape[0]; j++) {
    const trainImage = trainImages.slice(j, 1).flatten();
    const trainLabel = trainLabels.slice(j, 1).flatten();

    const distance = tf.sqrt(
      tf.sum(
        testImage.sub(trainImage).square()
      )
    );

    result.push({
      distance: distance.dataSync()[0],
      label: trainLabel.argMax(0).dataSync()[0],
    });

    // console.log(`distance = ${distance.dataSync()[0]}, label = ${trainLabel.argMax(0).dataSync()[0]}`);
  }

  const topK = lodash
    .sortBy(result, 'distance', o => o.distance)
    .slice(0, K);

  // console.log({ topK });

  const counter = lodash.countBy(topK, 'label');
  // console.log({ counter });

  const max = lodash.maxBy([...Object.entries(counter)], o => o[1]);
  const prediction = +max[0];
  const actualLabel = testLabel.argMax(0).dataSync()[0];

  // console.log({ predictLabel, actualLabel });

  if (prediction === actualLabel) {
    acc += 1 / testImages.shape[0];
  }

  predictions.push(prediction);
  labels.push(actualLabel);

  // if (i % 2 === 0) {
  printLog();
  // }
}

printLog();

function printLog() {
  const mse = tf.losses.meanSquaredError(
    tf.tensor(labels),
    tf.tensor(predictions),
  );
  console.log(`loss = ${mse.dataSync()[0]}, acc = ${acc.toFixed(3)}`);
}
