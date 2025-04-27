
**BioSight User Manual**

**1. Introduction**

Welcome to BioSight! This application allows you to upload biological images, automatically classify them into predefined categories ("Amphibia","Animalia",
    "Arachnida",
    "Aves",
    "Fungi",
    "Insecta",
    "Mammalia",
    "Mollusca",
    "Plantae",
    "Reptilia"), and organize them for easy viewing and management. You can also correct classifications and download your organized images.

**2. Getting Started**

*   **Accessing the Application:** Open your web browser and navigate to the application's main address (e.g., `http://localhost:8000`).
*   **Login:** If you are not already logged in, you will be automatically redirected to the Login page (#login.html).
    *   Enter the email address and password associated with your account.
    *   Click the "Login" button.
    *   Upon successful login, you will be taken to the main Upload page (#index.html).
    *   If login fails, an error message will appear below the relevant input field.
*   **Registration:** If you don't have an account, click the "Register" link on the Login page. This will take you to the Registration page (#register.html).
    *   Fill in your Full Name, Email address, desired Password, and confirm the password.
    *   Click the "Register" button.
    *   If registration is successful, you will be automatically logged in and redirected to the main Upload page (#index.html).
    *   If there are errors (e.g., passwords don't match, email already exists), messages will appear below the relevant input fields.

**3. Using the Application**

*   **The Home Page (Upload Interface - #index.html):**
    *   After logging in, you'll see the main page.
    *   Your name is displayed in the top navigation bar.
    *   The primary feature is the upload form.
*   **Uploading and Classifying Images:**
    1.  Click the "Select Images" or "Choose Files" button within the upload form (#index.html).
    2.  Select one or more image files from your computer. Allowed file types are `.png`, `.jpg`, and `.jpeg`.
    3.  Click the "Upload and Classify" button.
    4.  A loading overlay will appear indicating that the images are being processed.
    5.  Once processing is complete, the page will automatically refresh to display the Classification Results page (#result.html).

**4. Managing Results (#result.html)**

*   **Understanding the Results Page:** This page displays all the images you just uploaded, organized by their predicted biological class.
*   **Navigation:**
    *   **Download All as Zip:** If images were successfully processed, this button appears. Clicking it will download a `.zip` file containing all the currently organized images, sorted into folders by class.
    *   **Back to Upload:** Takes you back to the main upload page (#index.html).
    *   **Logout:** Logs you out of the application.
*   **Navigating Tabs:**
    *   Images are grouped into tabs based on their predicted class (e.g., "Mammalia", "Plantae", "unknown").
    *   Click on a tab button to view only the images classified under that category. The active tab is highlighted.
*   **Viewing Images:** Within each tab, images are displayed as cards. Each card shows:
    *   A preview of the image.
    *   The original filename.
    *   A delete button (×).
    *   A dropdown menu showing the current classification.
*   **Changing Image Classification:**
    1.  Locate the image card whose classification you want to change.
    2.  Click the dropdown menu below the image details.
    3.  Select the correct class from the list.
    4.  The application will automatically:
        *   Move the image file to the new class folder on the server.
        *   Update the image card's location, moving it to the correct tab on the results page (a new tab will be created if it doesn't exist).
        *   Update the image preview source.
        *   Add a star icon (★) next to the dropdown if the new class differs from the model's original prediction. This star is removed if you change it back to the original prediction.
        *   If the original tab becomes empty after moving the image, the tab and its button will be removed.
*   **Deleting Images:**
    1.  Click the red '×' button located at the top-right corner of the image preview you wish to delete.
    2.  A confirmation prompt will appear. Click "OK" to proceed.
    3.  The image card will be removed from the page.
    4.  The image file and its associated metadata will be deleted from the server.
    5.  If this was the last image in a tab, the tab and its button will also be removed.

**5. Logging Out**

*   Click the "Logout" button available on the main upload page (#index.html) or the results page (#result.html).
*   You will be redirected to the Login page (#login.html).

This covers the main functionalities available to a user in the BioSight application.