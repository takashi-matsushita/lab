def get_safe_balanced_split(target, train_ratio=0.8, get_test_indices=True, shuffle=False, seed=None):
  classes, counts = np.unique(target, return_counts=True)
  num_per_class = float(len(target))*float(train_ratio)/float(len(classes))
  if num_per_class > np.min(counts):
    print("Insufficient data to produce a balanced training data split.")
    print("Classes found {}".format(classes))
    print("Classes count {}".format(counts))
    ts = float(train_ratio*np.min(counts)*len(classes)) / float(len(target))
    print("train_ratio is reset from {} to {}".format(train_ratio, ts))
    train_ratio = ts
    num_per_class = float(len(target))*float(train_ratio)/float(len(classes))

  num_per_class = int(num_per_class)
  print("Data splitting on {} classes and returning {} per class".format(len(classes), num_per_class ))

  # get indices
  train_indices = []
  for c in classes:
    if seed is not None:
      np.random.seed(seed)
    c_idxs = np.where(target==c)[0]
    c_idxs = np.random.choice(c_idxs, num_per_class, replace=False)
    train_indices.extend(c_idxs)

  # get test indices
  test_indices = None
  if get_test_indices:
    test_indices = list(set(range(len(target))) - set(train_indices))

  # shuffle
  if shuffle:
    train_indices = random.shuffle(train_indices)
    if test_indices is not None:
      test_indices = random.shuffle(test_indices)

  return train_indices, test_indices
