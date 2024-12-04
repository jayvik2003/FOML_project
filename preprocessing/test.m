
% Extract the true labels (src) and predicted labels (t_out)
true_labels =src(:); % Flatten the array if necessary
predicted_labels = t_out(:); % Flatten the array if necessary

% Plot the confusion matrix
confusionchart(true_labels, predicted_labels, ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');

% Set title for the confusion matrix
title('Confusion Matrix');