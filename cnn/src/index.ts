import * as fs from 'fs';
import * as tf from '@tensorflow/tfjs-node';
import * as dataset from './dataset';
import createModel from './model';

// tf.enableProdMode();

function compileModel(model: tf.LayersModel) {
  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
}

async function train() {
  const [images, labels] = dataset.getTrainData();

  const model = createModel();
  compileModel(model);
  model.summary();

  const { history } = await model.fit(images, labels, {
    epochs: 5,
    batchSize: 32,
    validationSplit: 0.15,
  });

  await model.save('file://model');

  fs.writeFileSync(
    'model/history.json',
    JSON.stringify(history, null, 2),
    'utf8',
  );
}

async function test() {
  const [images, labels] = dataset.getTestData();

  const model = await tf.loadLayersModel('file://model/model.json');
  compileModel(model);

  const evalOutput = model.evaluate(images, labels) as tf.Scalar[];

  console.log(
    `\nEvaluation result:\n` +
    `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; ` +
    `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`,
  );

  // (model
  //   .predict(images.slice(0, 5)) as tf.Tensor<tf.Rank>)
  //   .argMax(1)
  //   .print();

  // labels.slice(0, 5)
  //   .argMax(1)
  //   .print();
}

train().then(test);

// tf.tensor([1, 2, 3, 4, 5, 6], [2, 3])
//   .mul(2)
//   .print();
