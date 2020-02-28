# ML_AI_HW_2
Machine Learning and Artificial Intelligence - Homework #2 - A.Y. 2018/2019 - Politecnico di Torino

### What to do 1⁄3 - Linear SVM
1. Load Iris dataset
2. Simply select the first two dimensions (let’s skip PCA this time)
3. Randomly split data into train, validation and test sets in proportion 5:2:3
4. For C from 10^(-3) to 10^3: (multiplying at each step by 10)
    - Train a linear SVM on the training set.
    - Plot the data and the decision boundaries
    - Evaluate the method on the validation set
5. Plot a graph showing how the accuracy on the validation set varies when
changing C
6. How do the boundaries change? Why?
7. Use the best value of C and evaluate the model on the test set. How well does
it go?

### What to do 2⁄3 - RBF Kernel
8. Repeat point 4. (train, plot, etc..), but this time use an RBF kernel
9. Evaluate the best C on the test set.
10. Are there any differences compared to the linear kernel? How are the
boundaries different?
11. Perform a grid search of the best parameters for an RBF kernel: we will now
tune both gamma and C at the same time. Select an appropriate range for
both parameters. Train the model and score it on the validation set.
12. Show the table showing how these parameters score on the validation set.
13. Evaluate the best parameters on the test set. Plot the decision boundaries.

### What to do 3/3 - K-Fold
14. Merge the training and validation split. You should now have 70% training and
30% test data.
15. Repeat the grid search for gamma and C but this time perform 5-fold
validation.
16. Evaluate the parameters on the test set. Is the final score different? Why?
