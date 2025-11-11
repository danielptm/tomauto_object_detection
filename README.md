## Tomauto and Yardguard object detection

#### TODO
1. Detect large path arrow
2. Detect small path arrow
3. Detect corners of arrows
4. Determine direction of arrow based on corners

### How to create yolo dataset
1. Create a labelme dataset with the same structure that is in `big_arrows/datasets/images`
Note: It should have the same file structure and the types of files per structure
2. Run `labelme2yolo --json_dir big_arrows/datasets/images/`
3. 