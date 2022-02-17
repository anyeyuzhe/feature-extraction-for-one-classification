A framework is proposed for one-class feature extraction. The proposed framework divides the original feature space into two orthogonal
spaces namely the principal space and the complementary space. The principal space is used to learn the features of the target class, and
the complementary space is used to learn the features of the abnormal class. The features that are extracted from the two spaces are
fused as the final extracted one-class feature of the original feature space. Furthermore, a specific implementation method, complete
principal component analysis (CPCA), is proposed. First, CPCA conducts principal component analysis (PCA) to calculate the projection
scores of the target class samples in the principal space. Then, according to the projection vectors of the principal components
(obtained in the principal space), the corresponding complementary space is constructed. The projection of the sample in the
complementary space is calculated and transformed into the first-order norm as the extracted feature in the complementary space. Dataset
of iris and warpAR10P are used to verify the effect of this proposed method. 
