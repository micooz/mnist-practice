import lodash from 'lodash';
import * as fs from 'fs';
import * as tf from '@tensorflow/tfjs-node';
import { DataSet, DataItem, FeatureMeta, Features } from './types';
import * as tree from './tree';
import { getTrainData, getTestData } from '../dataset';

const ALGORITHM = 'ID3';
const MODEL_PATH = `model/dt/model-${ALGORITHM}.json`;

// watermelon2.0 dataset
// async function getDataset(): Promise<{ dataset: DataSet; featureMetas: FeatureMeta[]; }> {
//   const csvDataset = tf.data.csv('file://src/dt/watermelon2.0.csv', {
//     columnConfigs: {
//       好瓜: {
//         isLabel: true,
//       },
//     },
//   });

//   const rawData = await csvDataset.toArray();

//   // @ts-ignore
//   const dataset = rawData.map(({ xs, ys }) => {
//     // skip first column
//     delete xs['编号'];

//     return {
//       features: xs,
//       label: Object.values(ys)[0],
//     } as DataItem;
//   });

//   return {
//     dataset,
//     featureMetas: [
//       { name: '色泽', type: 'discrete', enums: ['青绿', '乌黑', '浅白'] },
//       { name: '根蒂', type: 'discrete', enums: ['蜷缩', '稍蜷', '硬挺'] },
//       { name: '敲声', type: 'discrete', enums: ['浊响', '沉闷', '清脆'] },
//       { name: '纹理', type: 'discrete', enums: ['清晰', '稍糊', '模糊'] },
//       { name: '脐部', type: 'discrete', enums: ['凹陷', '稍凹', '平坦'] },
//       { name: '触感', type: 'discrete', enums: ['硬滑', '软粘'] },
//     ],
//   };
// }

const [trainImages, trainLabels] = getTrainData(200);
const [testImages, testLabels] = getTestData(2);

function prepareDataset(images: tf.Tensor<tf.Rank>, labels: tf.Tensor) {
  const dataset: DataSet = [];

  for (let i = 0; i < images.shape[0]; i++) {
    const image = images.slice(i, 1).flatten();
    const label = labels.slice(i, 1).flatten().argMax(0).dataSync()[0];

    dataset.push({
      features: [...image.dataSync().values()].reduce((acc, next, index) => {
        acc[index] = next;
        return acc;
      }, {} as Features),
      label,
    });
  }

  return {
    dataset,
    featureMetas: Object.keys(dataset[0].features).map(k => ({
      name: k,
      type: 'discrete',
      enums: lodash.range(0, 256),
    } as FeatureMeta)),
  };
}

function train() {
  const { dataset, featureMetas } = prepareDataset(trainImages, trainLabels);

  const model = tree.createTree(dataset, featureMetas, ALGORITHM);

  fs.writeFileSync(
    MODEL_PATH,
    JSON.stringify(model, null, 2),
    'utf8'
  );
}

function test() {
  const model = JSON.parse(
    fs.readFileSync(MODEL_PATH, 'utf8')
  );

  const { dataset } = prepareDataset(testImages, testLabels);

  let acc = 0;
  let correct = 0;

  dataset.forEach(({ features, label }, index) => {
    const result = tree.predict(model, features);

    if (result === label) {
      correct += 1;
      acc = correct / (index + 1);
    }

    console.log(`count = ${index + 1}  acc = ${acc}`);
  });
}

train();
test();
