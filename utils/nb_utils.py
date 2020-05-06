import seaborn as sns
import matplotlib.pyplot as plt
import os
import openslide

def plot_count(df, feature, title='', size=2):
    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))
    total = float(len(df))
    sns.countplot(df[feature],order = df[feature].value_counts().index, palette='Set2')
    plt.title(title)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show()



def plot_relative_distribution(df, feature, hue, title='', size=2):
    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))
    total = float(len(df))
    sns.countplot(x=feature, hue=hue, data=df, palette='Set2')
    plt.title(title)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show()

    
def display_masks(df, read_region=(0,0)):
    data = df
    f, ax = plt.subplots(3,3, figsize=(16,18))
    for i,data_row in enumerate(data.iterrows()):
        image = str(data_row[1][0])+'_mask.tiff'
        image_path = os.path.join(PATH,"train_label_masks",image)
        mask = openslide.OpenSlide(image_path)
        
        mask_data = mask.read_region(read_region, mask.level_count - 1, mask.level_dimensions[-1])
        cmap = matplotlib.colors.ListedColormap(['black', 'lightgray', 'darkgreen', 'yellow', 'orange', 'red'])
        ax[i//3, i%3].imshow(np.asarray(mask_data)[:,:,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5) 
        mask.close()       
        ax[i//3, i%3].axis('off')
        ax[i//3, i%3].axis('off')
        ax[i//3, i%3].set_title(f'ID: {data_row[1][0]}\nSource: {data_row[1][1]} ISUP: {data_row[1][2]} Gleason: {data_row[1][3]}')
        
    plt.show()
    
def print_slide_details(slide, show_thumbnail=True, max_size=(600,400)):
    """Print some basic information about a slide"""
    # Generate a small image thumbnail
    if show_thumbnail:
        display(slide.get_thumbnail(size=max_size))

    # Here we compute the "pixel spacing": the physical size of a pixel in the image.
    # OpenSlide gives the resolution in centimeters so we convert this to microns.
    spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)
    
    print(f"File id: {slide}")
    print(f"Dimensions: {slide.dimensions}")
    print(f"Microns per pixel / pixel spacing: {spacing:.3f}")
    print(f"Number of levels in the image: {slide.level_count}")
    print(f"Downsample factor per level: {slide.level_downsamples}")
    print(f"Dimensions of levels: {slide.level_dimensions}")


def display_patches(df, path, read_region=(1780,1950)):
    f, ax = plt.subplots(3,3, figsize=(16,18))
    
    for i,data_row in enumerate(df.iterrows()):
        # the image id is in the index, add tiff to it
        image = str(data_row[0])+'.tiff'
        
        # get the path
        image_path = os.path.join(path, image)
        image = openslide.OpenSlide(image_path)
        patch = image.read_region(read_region, 0, (256, 256))
        ax[i//3, i%3].imshow(patch) 
        image.close()       
        ax[i//3, i%3].axis('off')
        ax[i//3, i%3].set_title(f'ID: {data_row[0]}\nSource: {data_row[1][0]} ISUP: {data_row[1][1]} Gleason: {data_row[1][2]}')

    plt.show()
    
    
def pairwise_plot(df1, df2, path, max_size=(640,400)):
    ''' pairwise plot for karolinska and radboud'''
    assert df1.shape == df2.shape
    
    rows = df1.shape[0]
    
    f, ax = plt.subplots(rows,2, figsize=(16,18))
    
    for i, (kar_data, rad_data) in enumerate(zip(df1.iterrows(), df2.iterrows())):
        # get the image IDs
        image_id_kar = str(kar_data[0]) + '.tiff'
        image_id_rad = str(rad_data[0]) + '.tiff'

        # get the images = from the path
        image_path_kar = os.path.join(path, image_id_kar)
        image_path_rad = os.path.join(path, image_id_rad)

        # get the actual images
        image_kar = openslide.OpenSlide(image_path_kar)
        image_rad = openslide.OpenSlide(image_path_rad)

        # display the images
        ax[i, 0].imshow(image_kar.get_thumbnail(size=max_size))
        ax[i, 1].imshow(image_rad.get_thumbnail(size=max_size))

        # close
        image_rad.close()
        image_kar.close()

        ax[i,0].axis('off')
        ax[i,0].set_title(f'ID: {kar_data[0]}\nSource: {kar_data[1][0]} ISUP: {kar_data[1][1]} Gleason: {kar_data[1][2]}')

        ax[i,1].axis('off')
        ax[i,1].set_title(f'ID: {rad_data[0]}\nSource: {rad_data[1][0]} ISUP: {rad_data[1][1]} Gleason: {rad_data[1][2]}')
    plt.show()