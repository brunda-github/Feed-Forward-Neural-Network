def plot_ConfusionMatrix(y_pred, y_true, title):

  confusion_matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int32)
  for i in range(len(y_true)):
    confusion_matrix[y_true[i], y_pred[i]] += 1

  plt.figure(figsize=(10, 8))
  # Normalize the confusion matrix for better visualization
  norm = Normalize(vmin=confusion_matrix.min(), vmax=confusion_matrix.max())
  scaled_colors = norm(confusion_matrix)

  # Plot confusion matrix as heatmap with reversed 'Blues' colormap
  cmap = plt.cm.get_cmap('Blues_r')
  plt.imshow(scaled_colors, cmap=cmap, interpolation='nearest')

  # Add color bar
  sm = ScalarMappable(cmap=cmap, norm=norm)
  sm.set_array([])
  plt.colorbar(sm, ax=plt.gca())

  # Add class names as tick labels
  plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
  plt.yticks(np.arange(len(class_names)), class_names)

  # Add labels
  plt.title(title)
  plt.xlabel('Predicted label')
  plt.ylabel('True label')
  # Add numbers on top of each cell
  for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):
      plt.text(j, i, str(confusion_matrix[i, j]), ha='center', va='center', color='black')

  plt.savefig(title)