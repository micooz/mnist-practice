import lodash from 'lodash';
import { DataSet, DataValue, Features, FeatureMeta } from './types';
import * as utils from './utils';

type Node = {
  featureMeta?: FeatureMeta;
  branches?: Branch[];
  leaf?: true;
  label?: DataValue;
};

type Branch = {
  featureValue: DataValue;
  node: Node;
};

export function createTree(
  dataset: DataSet,
  featureMetas: FeatureMeta[],
  algorithm: 'ID3' | 'C45',
): Node {
  if (featureMetas.length === 0) {
    return createLeafNode(getMaxLabel(dataset));
  }

  // all data labels are the same
  const labelSet = new Set(dataset.map(item => item.label));
  if (labelSet.size === 1) {
    return createLeafNode(labelSet.values().next().value);
  }

  let gainResult;

  switch (algorithm) {
    case 'ID3':
      gainResult = divideByID3(dataset, featureMetas);
      break;
    case 'C45':
      gainResult = divideByC45(dataset, featureMetas);
      break;
    default:
      throw `unsupported algorithm: ${algorithm}`;
  }

  const { meta, result: { gain, gainRatio, dsMap } } = gainResult;

  const node: Node = {};
  node.featureMeta = meta;
  node.branches = [...dsMap.entries()].map(([featureValue, ds]) => {
    const newFeatureMetas =
      featureMetas.filter(item => item.name !== meta.name);

    return {
      featureValue,
      node: createTree(ds, newFeatureMetas, algorithm),
    };
  });

  // complement feature values
  const diff = lodash.difference(
    node.featureMeta.enums || [],
    node.branches.map(item => item.featureValue),
  );

  diff.forEach(value => {
    node.branches.push({
      featureValue: value,
      node: createLeafNode(getMaxLabel(dataset)),
    });
  });

  return node;
}

export function predict(tree: Node, features: Features): DataValue {
  const { featureMeta, branches, leaf, label } = tree;

  if (leaf) {
    return label;
  }

  const featureValue = features[featureMeta.name];
  if (!featureValue) {
    return null;
  }

  for (const branch of branches) {
    if (branch.featureValue === featureValue) {
      return predict(branch.node, features);
    }
  }

  return null;
}

function divideByID3(dataset: DataSet, featureMetas: FeatureMeta[]) {
  const ent = utils.entropy(dataset);

  const gainResults = featureMetas
    .map(meta => ({ meta, result: utils.gain(dataset, meta, ent) }));

  // sort by gain desc
  gainResults.sort((a, b) => b.result.gain - a.result.gain);

  // the max gain
  return gainResults[0];
}

function divideByC45(dataset: DataSet, featureMetas: FeatureMeta[]) {
  const ent = utils.entropy(dataset);

  const gainResults = featureMetas
    .map(meta => ({ meta, result: utils.gain(dataset, meta, ent) }));

  // sort by gain desc
  gainResults.sort((a, b) => b.result.gain - a.result.gain);

  // calculate the average gain
  const gainSum = lodash.sumBy(gainResults, item => item.result.gain);
  const gainAvg = gainSum / gainResults.length;

  return gainResults
    .filter(item => item.result.gain >= gainAvg)
    .sort((a, b) => b.result.gainRatio - a.result.gainRatio)[0];
}

function createLeafNode(label: DataValue): Node {
  return {
    leaf: true,
    label,
  };
}

function getMaxLabel(dataset: DataSet) {
  const map = new Map<DataValue, number>();

  dataset.forEach(({ label }) => {
    const count = map.get(label);
    if (!count) {
      map.set(label, 0);
    } else {
      map.set(label, count + 1);
    }
  });

  const result = [...map.entries()].sort((a, b) => {
    return b[1] - a[1];
  });

  return result[0][0];
}
