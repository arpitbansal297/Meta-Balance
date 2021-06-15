import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from PIL import Image

import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def l2_norm(input, axis = 1):
    # normalizes input with respect to second norm
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)

def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
    proportion = None,
    classes_to_demographic = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).
    Args:
        directory (str): root dataset directory
        class_to_idx (Optional[Dict[str, int]]): Dictionary mapping class name to class index. If omitted, is generated
            by :func:`find_classes`.
        extensions (optional): A list of allowed extensions.
            Either extensions or is_valid_file should be passed. Defaults to None.
        is_valid_file (optional): A function that takes path of a file
            and checks if the file is a valid file
            (used to check of corrupt files) both extensions and
            is_valid_file should not be passed. Defaults to None.
    Raises:
        ValueError: In case ``class_to_idx`` is empty.
        ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
        FileNotFoundError: In case no valid file was found for any class.
    Returns:
        List[Tuple[str, int]]: samples of a form (path_to_sample, class)
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    count = [0]*len(class_to_idx.keys())
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        # this shows the key is the directory and the mapping is our label
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index, 0, classes_to_demographic[target_class]
                    if proportion is None:
                        instances.append(item)
                    else:
                        if count[class_index] < proportion[class_index]:
                            instances.append(item)

                    count[class_index] += 1

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = available_classes - set(class_to_idx.keys())
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances



class ImageFolderWithProtectedAttributes(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithProtectedAttributes, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        young, male = int(path.split('_')[-3]), int(path.split('_')[-2])
        # make a new tuple that includes original and the path
        tuple_with_attr = (original_tuple + (young, male))
        #print(tuple_with_attr)
        return tuple_with_attr
    

class ImageFolderWithProtectedAttributesModify(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    def __init__(self, root, loader=default_loader, extensions=IMG_EXTENSIONS, transform=None,
                 target_transform=None, is_valid_file=None, gender_labels=None, proportions = None, classes_to_demographic = None):

        super(ImageFolderWithProtectedAttributesModify, self).__init__(root, transform=transform,
                                            target_transform=target_transform)

        #classes, class_to_idx = self._find_classes(self.root)
        classes, class_to_idx = self.classes, self.class_to_idx

        if classes_to_demographic == None:
            raise (RuntimeError("Mention the demographic"))

        # change classes and class_to_idx here
        if gender_labels is not None:
            new_to_old_labels = gender_labels['male'] + gender_labels['female']
            new_to_old_labels.sort()
            old_labels_to_new_labels = {}
            for i in range(len(new_to_old_labels)):
                old_labels_to_new_labels[new_to_old_labels[i]] = i

            new_dict = {}
            for folder in class_to_idx.keys():
                label = class_to_idx[folder]
                # check if this label is of my interest
                if label in new_to_old_labels:
                    # get the new label
                    nlabel = old_labels_to_new_labels[label]
                    # if yes map folder - new_label
                    new_dict[folder] = nlabel

            class_to_idx = new_dict

        classes = class_to_idx.keys() # original classes not index

        #####
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file, proportions, classes_to_demographic)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                                                                 "Supported extensions are: " + ",".join(
                extensions)))

        self.loader = default_loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = self.samples
        self.transform = transform

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        #original_tuple = super(ImageFolderWithProtectedAttributesModify, self).__getitem__(index)
        path, target, young, male = self.samples[index]
        img = Image.open(path)
        img.convert('RGB')
        img = self.transform(img)
        original_tuple = img, target, young, male

        return original_tuple



def evaluate(dataloader, criterion, backbone, head, emb_size,  k_accuracy = False, multilabel_accuracy = False):

    loss_male = 0
    loss_female = 0
    loss_overall = 0
    acc_male = 0
    acc_female = 0
    acc_overall = 0
    acc_k_male = 0
    acc_k_female = 0
    acc_k_overall = 0
    count_all = 0
    count_male = 0
    count_female = 0
    backbone.eval()
    head.eval()

    feature_matrix = torch.empty(0, emb_size)
    labels_all = []
    gender_all = []

    for inputs, labels, young, male in tqdm(iter(dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device).long()

        with torch.no_grad():

            if multilabel_accuracy:
                features = backbone(inputs)
                outputs = head(features, labels)
                loss = criterion(outputs, labels)

                # add sum of losses for female and male images
                loss_male += loss[male == 1].sum()
                loss_female += loss[male == -1].sum()
                loss_overall += loss.sum()

                # multiclass accuracy
                _, predicted = outputs.max(1)
                acc_overall += predicted.eq(labels).sum().item()
                acc_male += predicted[male == 1].eq(labels[male == 1]).sum().item()
                acc_female += predicted[male == -1].eq(labels[male == -1]).sum().item()
                count_all += inputs.shape[0]
                count_male += sum(male == 1)
                count_female += sum(male == -1)

            if k_accuracy:
                '''need to build feature matrix'''
                inputs_flipped = torch.flip(inputs, [3])
                embed = backbone(inputs) + backbone(inputs_flipped)
                features_batch = l2_norm(embed)
                feature_matrix = torch.cat((feature_matrix, features_batch.detach().cpu()), dim = 0)

                labels_all = labels_all + labels.cpu().tolist()
                gender_all = gender_all + male.cpu().tolist()


    if multilabel_accuracy:
        acc_overall = acc_overall/count_all
        acc_male = acc_male/count_male
        acc_female = acc_female/count_female

        loss_overall = loss_overall/count_all
        loss_male = loss_male/count_male
        loss_female = loss_female/count_female

    if k_accuracy:
        dist_matrix =  l2_dist(feature_matrix)
        acc_k_male, acc_k_female, acc_k_overall = predictions(dist_matrix, torch.tensor(labels_all), torch.tensor(gender_all))

    return loss_overall, loss_male, loss_female, acc_overall, acc_male, acc_female, acc_k_overall, acc_k_male, acc_k_female



def l2_dist(feature_matrix):
    ''' computing distance matrix '''
    return torch.cdist(feature_matrix, feature_matrix)

def predictions(dist_matrix, labels, male):
    nearest_neighbors = torch.topk(dist_matrix, dim=1, k = 2, largest = False)[1][:,1]
    n_images = dist_matrix.shape[0]
    correct = torch.zeros(labels.shape)
    for img in range(n_images):
        nearest_label = labels[nearest_neighbors[img]].item()
        if labels[img] == nearest_label:
            correct[img] = 1

    acc_k_male = (correct[male == 1]).mean()
    acc_k_female = (correct[male == -1]).mean()
    acc_k_overall = correct.mean()
    return acc_k_male, acc_k_female, acc_k_overall


