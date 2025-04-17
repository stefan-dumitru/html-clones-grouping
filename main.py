import os
import shutil
import numpy as np
import asyncio
from pathlib import Path
from sklearn.cluster import DBSCAN
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from collections import defaultdict
from PIL import Image
from playwright.async_api import async_playwright

# load the resnet50 model to extract features
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

# get all html files from the main directory
def get_html_files(directory):
    return [str(file) for file in Path(directory).glob("*.html")]

async def capture_screenshots(base_directory, output_base_dir):
    MAX_CONCURRENT_PAGES = 4  # how many browser pages are open at once (could be modified based on computer performance)
    MAX_RETRIES = 2  # try again if the screenshot process fails
    TIMEOUT_MS = 60000

    os.makedirs(output_base_dir, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context()
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_PAGES)  # prevents opening too many pages in parallel

        async def capture_with_retry(html_file, output_dir):
            filename = Path(html_file).stem + ".png"
            screenshot_path = os.path.join(output_dir, filename)

            for attempt in range(1, MAX_RETRIES + 2):
                try:
                    async with semaphore:
                        page = await context.new_page()
                        await page.goto(f"file://{html_file}", wait_until="domcontentloaded", timeout=TIMEOUT_MS)
                        await page.screenshot(path=screenshot_path, full_page=True)  # take screenshot of the page
                        await page.close()
                        print(f"Captured: {screenshot_path}")
                        return
                except Exception as e:
                    print(f"Attempt {attempt} failed for {html_file}: {e}")
                    if attempt == MAX_RETRIES + 1:
                        print(f"Skipping {html_file} after {MAX_RETRIES + 1} attempts")

        for subfolder in Path(base_directory).iterdir():
            if subfolder.is_dir():
                html_files = get_html_files(subfolder)
                if not html_files:
                    continue

                output_dir = os.path.join(output_base_dir, subfolder.name)
                os.makedirs(output_dir, exist_ok=True)

                tasks = [capture_with_retry(html_file, output_dir) for html_file in html_files]
                await asyncio.gather(*tasks)

        await browser.close()

# resize and crop high-res screenshots to 224x224 while preserving layout visibility
def custom_preprocess_high_res_image(img_path):
    img = Image.open(img_path).convert("RGB")
    width, height = img.size

    # resize while preserving aspect ratio to 224 width
    img = img.resize((224, int(height * (224 / width))), Image.BILINEAR)

    # crop to get exactly 224x224
    if img.height >= 224:
        img = img.crop((0, 0, 224, 224))
    else:
        new_img = Image.new("RGB", (224, 224), (0, 0, 0))
        new_img.paste(img, (0, (224 - img.height) // 2))
        img = new_img

    img_array = image.img_to_array(img)
    return np.expand_dims(img_array, axis=0)

# extract resnet50 features for each image in batches
def batch_extract_features(image_paths, batch_size=16):
    features = []
    for i in range(0, len(image_paths), batch_size):
        batch_images = []
        for img_path in image_paths[i:i + batch_size]:
            with Image.open(img_path) as img:
                width, height = img.size

            # apply custom preprocessing to high-res screenshots
            if width == 1280 and 5900 <= height <= 6700:
                img_array = custom_preprocess_high_res_image(img_path)
            else:
                # same preprocessing for small resolution files
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)

            batch_images.append(img_array)

        batch_images = np.vstack(batch_images)
        batch_images = preprocess_input(batch_images)
        batch_features = model.predict(batch_images, batch_size=batch_size, verbose=0)
        features.extend(batch_features)

    return np.array(features)

# cluster screenshots using dbscan
def group_html_by_visual_similarity(screenshot_dir, output_clusters_dir):
    os.makedirs(output_clusters_dir, exist_ok=True)

    for subfolder in Path(screenshot_dir).iterdir():
        if not subfolder.is_dir():
            continue

        screenshot_files = list(Path(subfolder).glob("*.png"))
        if not screenshot_files:
            continue

        image_paths = [str(screenshot) for screenshot in screenshot_files]
        features = batch_extract_features(image_paths, batch_size=16)

        # dbscan clustering
        dbscan = DBSCAN(eps=10, min_samples=2, metric='euclidean')
        cluster_labels = dbscan.fit_predict(features)

        # group by dbscan labels
        clusters = defaultdict(list)
        noise_points = []  # for htmls with no pairings

        for img_path, cluster_id in zip(image_paths, cluster_labels):
            if cluster_id == -1:
                noise_points.append(img_path)
            else:
                clusters[cluster_id].append(img_path)

        # assign an unique cluster id to each noise point
        next_cluster_id = max(clusters.keys(), default=-1) + 1
        for img_path in noise_points:
            clusters[next_cluster_id] = [img_path]
            next_cluster_id += 1

        # ensure cluster ids are indexed starting from 0 and are sequential
        sorted_cluster_items = sorted(clusters.items())  # sorted by original cluster id
        reindexed_clusters = {new_id: files for new_id, (_, files) in enumerate(sorted_cluster_items)}

        # print results
        print(f"\nSubdirectory: {subfolder.name}")
        cluster_results = [f"[{', '.join(Path(f).stem + '.html' for f in files)}]" for files in reindexed_clusters.values()]
        print(",\n".join(cluster_results))

        # save results in folders
        subfolder_output = Path(output_clusters_dir) / subfolder.name
        if subfolder_output.exists():
            shutil.rmtree(subfolder_output)
        os.makedirs(subfolder_output, exist_ok=True)

        for cluster_id, files in reindexed_clusters.items():
            cluster_folder = subfolder_output / f"Cluster_{cluster_id}"
            os.makedirs(cluster_folder, exist_ok=True)

            for screenshot_file in files:
                destination_path = cluster_folder / Path(screenshot_file).name
                if not destination_path.exists():
                    shutil.copy(screenshot_file, destination_path)

# paths
project_root = Path(__file__).parent.resolve()

base_directory = project_root / "clones"
output_screenshots_dir = project_root / "screenshots"
output_clusters_dir = project_root / "clusters"

# run
asyncio.run(capture_screenshots(base_directory, output_screenshots_dir))
group_html_by_visual_similarity(output_screenshots_dir, output_clusters_dir)