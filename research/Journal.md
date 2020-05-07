### Plans
1. Determine the similarity between the cropped faces
    * Use face_recoginition python module


----------
07.05.2020







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