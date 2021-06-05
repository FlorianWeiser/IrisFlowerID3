import csv
import random
import math

# stopping criterion
MINIMUM_SAMPLE_SIZE = 4
MAX_TREE_DEPTH = 3


class tree_node:

    def __init__(self, training_set, attribute_list, attribute_values, tree_depth):
        self.is_leaf = False
        self.dataset = training_set
        self.split_attribute = None
        self.split = None
        self.attribute_list = attribute_list
        self.attribute_values = attribute_values
        self.left_child = None
        self.right_child = None
        self.prediction = None
        self.depth = tree_depth

    def build(self):

        training_set = self.dataset

        # only proceed building tree if stopping criterion isn't matched
        # (number of tuples below threshold or all instances belong to same class)
        if self.depth < MAX_TREE_DEPTH and len(training_set) >= MINIMUM_SAMPLE_SIZE and len(set([elem["species"] for elem in training_set])) > 1:
            # get attribute and split with highest information gain
            max_gain, attribute, split = max_information_gain(self.attribute_list, self.attribute_values, training_set)

            # test if information gain is greater than 0 (another stopping criterion)
            if max_gain > 0:
                # split tree
                self.split = split
                self.split_attribute = attribute

                # create chrildren
                training_set_l = [elem for elem in training_set if elem[attribute] < split]
                training_set_r = [elem for elem in training_set if elem[attribute] >= split]
                self.left_child = tree_node(training_set_l, self.attribute_list, self.attribute_values, self.depth + 1)
                self.right_child = tree_node(training_set_r, self.attribute_list, self.attribute_values, self.depth + 1)
                self.left_child.build()
                self.right_child.build()
            else:
                self.is_leaf = True
        else:
            self.is_leaf = True

        if self.is_leaf:
            # prediction of leaf is the most common class in training_set
            setosa_count = versicolor_count = virginica_count = 0
            for elem in training_set:
                if elem["species"] == "Iris-setosa":
                    setosa_count += 1
                elif elem["species"] == "Iris-versicolor":
                    versicolor_count += 1
                else:
                    virginica_count += 1
            dominant_class = "Iris-setosa"
            dom_class_count = setosa_count
            if versicolor_count >= dom_class_count:
                dom_class_count = versicolor_count
                dominant_class = "Iris-versicolor"
            if virginica_count >= dom_class_count:
                dom_class_count = virginica_count
                dominant_class = "Iris-virginica"
            self.prediction = dominant_class

    # test decision tree accuracy
    def predict(self, sample):
        if self.is_leaf:
            return self.prediction
        else:
            if sample[self.split_attribute] < self.split:
                return self.left_child.predict(sample)
            else:
                return self.right_child.predict(sample)


    # this isn't pruning, this just merges two leaves if they have the same precition
    def merge_leaves(self):
        if not self.is_leaf:
            self.left_child.merge_leaves()
            self.right_child.merge_leaves()
            if self.left_child.is_leaf and self.right_child.is_leaf and self.left_child.prediction == self.right_child.prediction:
                self.is_leaf = True
                self.prediction = self.left_child.prediction


    def print(self, prefix):
        if self.is_leaf:
            print("\t" * self.depth + prefix + self.prediction)
        else:
            print("\t" * self.depth + prefix + self.split_attribute + "<" + str(self.split) + "?")
            self.left_child.print("[True] ")
            self.right_child.print("[False] ")


class ID3_tree:
    def __init__(self):
        self.root = None

    def build(self, training_set, attribute_list, attribute_values):
        self.root = tree_node(training_set, attribute_list, attribute_values, 0)
        self.root.build()

    def merge_leaves(self):
        self.root.merge_leaves()

    def predict(self, sample):
        return self.root.predict(sample)

    def print(self):
        print("----------------")
        print("DECISION TREE")
        self.root.print("")
        print("----------------")

# calculate the entropy of a target_attribute for a given set
# in our example, the target attribute is species and valid values are Iris-setosa, Iris-versicolor and Iris-virginica
def entropy(dataset):

    if len(dataset) == 0:
        return 0

    target_attribute_name = "species"
    target_attribute_values = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    data_entropy = 0
    for val in target_attribute_values:

        # calculate the probability p that an element in the set has the value val
        p = len([elem for elem in dataset if elem[target_attribute_name] == val]) / len(dataset)

        if p > 0:
            data_entropy += -p * math.log(p, 2)

    return data_entropy


# calculate average entropy of split on an attribute
# Split is the binary split limit for the attribute
def info_gain(attribute_name, split, dataset):

    # split set and calculate probabilities that elements are in the splits
    set_smaller = [elem for elem in dataset if elem[attribute_name] < split]
    p_smaller = len(set_smaller) / len(dataset)
    set_greater_equals = [elem for elem in dataset if elem[attribute_name] >= split]
    p_greater_equals = len(set_greater_equals) / len(dataset)

    # calculate information gain
    info_gain = entropy(dataset)
    info_gain -= p_smaller * entropy(set_smaller)
    info_gain -= p_greater_equals * entropy(set_greater_equals)

    return info_gain


# get criterion and optimal split to maximize information gain
def max_information_gain(attribute_list, attribute_values, dataset):

    max_info_gain = 0
    for attribute in attribute_list:  # test all input attributes
        for split in attribute_values[attribute]:  # test all possible values as split limits
            split_info_gain = info_gain(attribute, split, dataset)  # calculate information gain
            if split_info_gain >= max_info_gain:
                max_info_gain = split_info_gain
                max_info_gain_attribute = attribute
                max_info_gain_split = split
    return max_info_gain, max_info_gain_attribute, max_info_gain_split


def read_iris_dataset():
    dataset = []
    with open('IRIS.csv', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        is_first = True
        for row in reader:

            instance = {}
            if not is_first:
                # skip first row
                instance["sepal_length"] = float(row[0])
                instance["sepal_width"] = float(row[1])
                instance["petal_length"] = float(row[2])
                instance["petal_width"] = float(row[3])
                instance["species"] = row[4]
                dataset.append(instance)
            is_first = False

    return dataset


if __name__ == '__main__':

    # load dataset from csv file
    dataset = read_iris_dataset()

    if not dataset:
        # dataset is empty
        print('dataset is empty!')
        exit(1)

    # choose random 25% of list as sample
    test_set = random.sample(dataset, int(0.25 * len(dataset)))
    test_set_dupl = test_set.copy()  # only needed to generate training set
    training_set = [i for i in dataset if not i in test_set_dupl or test_set_dupl.remove(i)]
    print('dataset size:', len(dataset))
    print('training set size:', len(training_set))
    print('test set size:', len(test_set))

    # list of all input attributes
    attr_list = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    # get list of all valid attribute values
    # this will later be needed to calculate the information gain
    attr_domains = {}
    for attr in list(dataset[0].keys()):
        attr_domain = set()
        for s in dataset:
            attr_domain.add(s[attr])
        attr_domains[attr] = list(attr_domain)

    # build decision tree
    dt = ID3_tree()
    dt.build(dataset, attr_list, attr_domains)
    dt.merge_leaves()  # merge leaves with the same prediction

    # calculate accuracy with test set
    accuracy = 0
    for sample in test_set:
        if sample["species"] == dt.predict(sample):
            accuracy += (1/len(test_set))

    dt.print()

    print("accuracy on test set: " + "{:.2f}".format(accuracy * 100) + "%")