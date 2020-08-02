import * as assert from 'assert';
import * as fs from 'fs';
import * as tf from '@tensorflow/tfjs-node';
import { DataSet, DataItem } from './types';
import * as tree from './tree';

const MODEL_PATH = 'model/dt/model.json';

async function getDataset(): Promise<{
  dataset: DataSet;
  featureNames: string[];
}> {
  const csvDataset = tf.data.csv('file://src/dt/watermelon2.0.csv', {
    columnConfigs: {
      好瓜: {
        isLabel: true,
      },
    },
  });

  const rawData = await csvDataset.toArray();
  const columnNames = await csvDataset.columnNames();

  // @ts-ignore
  const dataset = rawData.map(({ xs, ys }) => {
    // skip first column
    delete xs['编号'];

    return {
      features: xs,
      label: Object.values(ys)[0],
    } as DataItem;
  });

  return {
    dataset,
    featureNames: columnNames.slice(1, -1),
  };
}

async function train() {
  const { dataset, featureNames } = await getDataset();
  // console.log(dataset);

  const model = tree.createTree(dataset, featureNames);

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

  const features = {
    色泽: '青绿',
    // 色泽: '浅白',
    根蒂: '稍蜷',
    敲声: '浊响',
    纹理: '清晰',
    脐部: '稍凹',
    触感: '软粘',
  };

  const result = tree.predict(model, features);
  console.log(result);

  assert.equal(result, '是');
}

train().then(test);

// test();
