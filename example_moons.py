from sklearn.datasets import make_moons
from CMS import CMS, AutoLinearPolicy

# Generate moons data set
x, y = make_moons(shuffle=False, noise=.01)
# Create one cannot-link constraint from center of one moon to another
cl = [[25, 75]]

# Create bandwidth policy as used in our experiments
pol = AutoLinearPolicy(x, 100)
# Use nonblurring mean shift (do not move sampling points)
cms = CMS(pol, max_iterations=100, blurring=False, label_merge_k=.999)
cms.fit(x, cl)

# Plot the result
from CMS.Plotting import plot_clustering
import matplotlib.pyplot as plt

plot_clustering(x, cms.labels_, cms.modes_, cl=cl)

plt.show()
