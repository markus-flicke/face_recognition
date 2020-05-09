### Iterative training
* Give the first picture a class
* For the next picture
    * What is the most similar picture?
    * What is your confidence
    * If confidence is high, assign the predicted label
    * If confidence is low, assign a new label
* Repeat



### ToDos
* Don't write the training set to disk all the time



----------
09.05.2020
* Eigenfaces and Fisherfaces are not easy to train
    * They expect square images
* Testing
    * Mediocre results for lbd_hog, as accuracy is roughly 35%
        * Actually removing duplicates was not fully successful
        * Hence the accuracy should be even lower
            * Maybe 10%
    * Leave one out CV
        * Can only be done with people that appear at least twice
            * And that don't appear twice on the same picture
                * The legacy code occasionally yields duplicates
            * Introduced a cleaning step that removes duplicates based on labels

----------
08.05.2020
* Wrote an Excel file for labelling
    * Three columns
        * filename, image, label

----------
07.05.2020
* Vertical test of Local Binary Patterns with Histogram Oriented Gradients
    * Successfully predicted 2/2
* Needed to install opencv as follows:
    * `pip3 install opencv-contrib-python`


----------
03.05.2020
* Crop faces from pictures
    * Use FotoAlbumExtractor
    * If intial results are not so great, attempt to increase cropping region
        * Could make things worse because of more information to process
* Mr. Geitgey's face recognition [project](https://github.com/ageitgey/face_recognition)
    * Provides some face recognition
    * Last updates in February 2020 -> fairly current
    * Works mediocrely on our greyscale images
        * Mother is recognised in group, but other people are also classified as mother/ aunt
            * Maybe just more training data would be required
* FotoAlbumExtractor
    * Clips photos from the album
    * Marks faces with rectangles
        * I propose to crop faces to then use the face_recognition module
    * Found a little bug in backgroundremover.py
        * Function returns two variables, 3 expected
        * This was probably some update to the cv2 library. Don't think they would have submitted code that doesn't run.
    * I incorporated their project 
        * Had to remove the config file
        * Started using their photo extractor