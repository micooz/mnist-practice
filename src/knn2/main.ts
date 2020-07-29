import * as knnClassifier from '@tensorflow-models/knn-classifier';
import { getTrainData, getTestData } from '../dataset';

const classifier = knnClassifier.create();

const [trainImages, trainLabels] = getTrainData(1000);
const [testImages, testLabels] = getTestData();

for (let i = 0; i < trainImages.shape[0]; i++) {
  const trainImage = trainImages.slice(i, 1).flatten();
  const trainLabel = trainLabels.slice(i, 1).flatten().argMax(0).dataSync()[0];

  classifier.addExample(
    trainImage,
    trainLabel,
  );
}

async function main() {
  let acc = '0.000';
  let correctNum = 0;
  for (let i = 0; i < testImages.shape[0]; i++) {
    const testImage = testImages.slice(i, 1).flatten();
    const testLabel = testLabels.slice(i, 1).flatten().argMax(0).dataSync()[0];

    const result = await classifier.predictClass(
      testImage,
      5,
    );

    const { label, classIndex, confidences } = result;
    if (testLabel === +label) {
      correctNum += 1;
      acc = (correctNum / (i + 1)).toFixed(3);
    }
    if ((i + 1) % 100 === 0) {
      console.log(`total = ${i + 1}  acc = ${acc}`);
    }
  }
}

main();
