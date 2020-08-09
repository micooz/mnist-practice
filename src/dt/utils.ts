import lodash from 'lodash';
import { DataSet, DataValue, FeatureMeta } from './types';

// information entropy
export function entropy(dataset: DataSet) {
  return -lodash.sumBy(pks(dataset), pk => pk * Math.log2(pk));
}

// gini value
export function gini(dataset: DataSet) {
  return 1 - lodash.sumBy(pks(dataset), pk => pk * pk);
}

// information gain (ID3 & C4.5)
export function gain(dataset: DataSet, featureMeta: FeatureMeta, prevEnt: number) {
  const dsMap = divide(dataset, featureMeta);

  const [ent, iv] = [...dsMap.values()].reduce((acc, next) => {
    const weight = next.length / dataset.length;
    acc[0] += weight * entropy(next); // ID3
    acc[1] -= weight * Math.log2(weight); // C4.5
    return acc;
  }, [0, 0]);

  const gainValue = prevEnt - ent;

  return {
    gain: gainValue,
    gainRatio: gainValue / iv, // intrinsic value
    dsMap,
  };
}

// gini index (CART)
export function giniIndex(dataset: DataSet, featureMeta: FeatureMeta) {
  const dsMap = divide(dataset, featureMeta);

  const giniIndex = [...dsMap.values()].reduce((acc, next) => {
    const weight = next.length / dataset.length;
    acc += weight * gini(next);
    return acc;
  }, 0);

  return {
    giniIndex,
    dsMap,
  };
}

function pks(dataset: DataSet) {
  const kMap = new Map<DataValue, number>();

  for (let i = 0; i < dataset.length; i++) {
    const { label } = dataset[i];
    const count = kMap.get(label) | 0;
    if (!count) {
      kMap.set(label, 0);
    }
    kMap.set(label, count + 1);
  }

  return [...kMap.values()].map(count => count / dataset.length);
}

function divide(dataset: DataSet, featureMeta: FeatureMeta) {
  const dsMap = new Map<DataValue, DataSet>();
  const { name, type } = featureMeta;


  for (const item of dataset) {
    const value = item.features[name];

    const ds = dsMap.get(value);
    if (!ds) {
      dsMap.set(value, [item]);
    } else {
      ds.push(item);
    }
  }

  return dsMap;
}
