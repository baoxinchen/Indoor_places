import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

RGB_FOLDER = '_RGB'
SEG_FOLDER = '_seg_img'


def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    
    #img = Image.open(filename)
    #imgl = Image.open(filename.replace(RGB_FOLDER, SEG_FOLDER).replace('jpg','png'))
    
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS) #and img.mode == 'RGB' and imgl.mode == 'L'


def find_classes(dir, classifier_type):
    
    file_name = './Taxonomy.txt'
    classes = list()
    with open(file_name) as class_file:
        reading = 'off'
        for line in class_file:
            if line.lower().startswith(classifier_type.lower()): reading = 'on'
            elif line.lower().startswith('end'): reading = 'off'
            if reading == 'on' and '/' in line:
                classes.append(line.split(' ')[0].replace('|-','').replace(',',''))
    #classes.sort()
    #classes = [classes[364]]
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(maindir, class_to_idx, classifier_type):
    images = []
    #maindir = os.path.expanduser(maindir)
    
    file_name = './Taxonomy.txt'
    classes = list()
    with open(file_name) as class_file:
        reading = 'off'
        for line in class_file:
            if line.lower().startswith(classifier_type.lower()): reading = 'on'
            elif line.lower().startswith('end'): reading = 'off'
            if reading == 'on' and '/' in line:
                class_pathes=list()
                for s in line.split(' '):
                    if s.startswith('/'):class_pathes.append(s[1:])
                classes.append([line.split(' ')[0].replace('|-','').replace(',',''), class_pathes])
    #classes.sort()
    #classes = [classes[364]]
    #print classes
    for this_class in classes:
        target = this_class[0]
        for this_path in this_class[1]:
            d = os.path.join(maindir, this_path)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_image_file(path):
                        item = (path, class_to_idx[target])
                        images.append(item)

    return images


def pil_loader(path, num_channels):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
    if num_channels == 4:
        with open(path.replace(RGB_FOLDER, SEG_FOLDER).replace('jpg','png'),'rb') as f:
            img_seg = Image.open(f)
            img_seg = img_seg.convert('L').resize(img.size, resample=Image.NEAREST)
            #print path.replace(RGB_FOLDER, SEG_FOLDER).replace('jpg','png')
            img.putalpha(img_seg)
    return img


def accimage_loader(path, num_channels):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path, num_channels)


def default_loader(path, num_channels):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path, num_channels)
    else:
        return pil_loader(path, num_channels)



class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, classifier_type='Home', num_channels=4):
        
        classes, class_to_idx = find_classes(root, classifier_type)
        imgs = make_dataset(root, class_to_idx, classifier_type)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        print(len(classes))
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.num_channels = num_channels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path, self.num_channels)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        #print(target)
        #print img 
        return img, target

    def __len__(self):
        return len(self.imgs)
    
