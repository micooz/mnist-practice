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

export function createTree(dataset: DataSet, featureMetas: FeatureMeta[]): Node {
  if (featureMetas.length === 0) {
    return createLeafNode(getMaxLabel(dataset));
  }

  // all data labels are the same
  const labelSet = new Set(dataset.map(item => item.label));
  if (labelSet.size === 1) {
    return createLeafNode(labelSet.values().next().value);
  }

  const ent = utils.entropy(dataset);
  // console.log(ent);

  let maxGain = 0;
  let resultFeatureMeta: FeatureMeta = null;
  let resultDsMap: Map<DataValue, DataSet> = new Map();

  // find best divide point
  featureMetas.forEach(meta => {
    const { value, dsMap } = utils.gain(dataset, meta, ent);

    if (value > maxGain) {
      maxGain = value;
      resultFeatureMeta = meta;
      resultDsMap = dsMap;
    }
  });

  const node: Node = {};
  node.featureMeta = resultFeatureMeta;
  node.branches = [...resultDsMap.entries()].map(([featureValue, ds]) => {
    const newFeatureMetas =
      featureMetas.filter(meta => meta.name !== resultFeatureMeta.name);

    return {
      featureValue,
      node: createTree(ds, newFeatureMetas),
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
