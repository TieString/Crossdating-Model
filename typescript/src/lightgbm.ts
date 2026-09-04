export interface LightGbmLeaf {
  leaf_index: number;
  leaf_value: number;
}

export interface LightGbmSplit {
  split_index: number;
  split_feature: number;
  threshold: number | string;
  decision_type: string;
  default_left: boolean;
  missing_type: string;
  left_child: LightGbmNode;
  right_child: LightGbmNode;
}

export type LightGbmNode = LightGbmLeaf | LightGbmSplit;

export interface UnifiedV5Model {
  columns: string[];
  identities: [string, number][];
  eventGate: number;
  windowWidth: number;
  treeInfo: { tree_structure: LightGbmNode }[];
}

function isLeaf(node: LightGbmNode): node is LightGbmLeaf {
  return "leaf_value" in node;
}

function scoreNode(node: LightGbmNode, features: readonly number[]): number {
  let current = node;
  while (!isLeaf(current)) {
    if (current.decision_type !== "<=") {
      throw new Error(`unsupported LightGBM decision type: ${current.decision_type}`);
    }
    const value = features[current.split_feature];
    const missing = value === undefined || Number.isNaN(value);
    const left = missing ? current.default_left : value <= Number(current.threshold);
    current = left ? current.left_child : current.right_child;
  }
  return current.leaf_value;
}

export function scoreCandidate(model: UnifiedV5Model, features: readonly number[]): number {
  if (features.length !== model.columns.length) {
    throw new Error(`expected ${model.columns.length} features, received ${features.length}`);
  }
  let score = 0;
  for (const tree of model.treeInfo) score += scoreNode(tree.tree_structure, features);
  return score;
}

export function scoreCandidates(model: UnifiedV5Model, rows: readonly (readonly number[])[]): number[] {
  return rows.map((row) => scoreCandidate(model, row));
}
