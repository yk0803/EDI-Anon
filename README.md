# EDI-Anon
Explainability-Driven Iterative Image Anonymization

Abstract: Human face images require anonymization before being shared publicly for secondary use, safeguarding against GDPR violations, and respecting individuals' privacy. However, current image anonymization methods severely hamper the analytical utility of the protected images. In this paper, we address the challenge of balancing privacy protection and utility preservation in image anonymization. We propose a general framework that significantly improves existing image anonymization methods. To achieve this, we introduce a disclosure risk-aware approach that leverages explainability techniques to target identity-revealing features within the images. Contrary to conventional methods, which uniformly perturb image pixels, our proposal focuses on perturbing those pixels that contribute the most to disclosure. Moreover, pixel perturbation is enforced incrementally, and it is driven by the observed residual risk. The versatility of our framework extends to supporting a wide variety of standard perturbation techniques as its core, allowing flexibility in selecting the most suitable approach for anonymization. Empirical results demonstrate that, even with the simplest perturbation techniques, our approach leads to a significantly improved trade-off between privacy protection and utility preservation compared to the current state-of-the-art.

# The Dataset
We have used the MTF data set which contains 5246 images with a distinct distribution of celebrities' image faces that emerged across different labels. While our initial efforts aimed to crawl a balanced number of images across all tasks and labels, the real distribution of the data available online led to an imbalance within the data set. This imbalance can be attributed to various factors. I) Celebrities from different regions of the world publish their images at varying rates and under different copyright licenses. II) Young celebrities tend to publish their images more frequently than elderly celebrities. III) Elderly celebrities often have more images from their younger days than images from their current age.

# Get the Dataset
The MTF data set can be accessed through the following link

https://sobigdata.d4science.org/catalogue-sobigdata?path=/dataset/multi-task_faces_mtf_dataset

The second version of the data set for single tasks and the trained models are available on Google Drive through the following link: 
https://drive.google.com/drive/folders/1FCSCaBMkGZ6GFcOHmfbFGPcgucRaeCrf?usp=sharing.

Researchers can conveniently access the data for each task from this version, which streamlines their workflow and simplifies experimentation.

Moreover, we also made available all the trained models we evaluated on the MTF data set. Since these models provide baseline results for the different tasks supported by the data set, by releasing them we aim at facilitating future investigations conducted on this data set.

https://drive.google.com/drive/folders/1PCWnpapplpvfNXqBs5acTqL0nfVuM5uO?usp=drive_link

To Execute the code you can follow these steps

# 1. Install the Required packages and libraries
      Provided in the Requirements.txt file
# 2. The simple codes folder
      This folder contains the original codes of the simple techniques, saved with their respective names for ease of access
# 3. For the Proposed Method
      After you have downloaded the datasets and the models, and have installed the required libraries & packages,
      1. Open the ipynb file named Separated (for Masking use the Masking New ipynb file)
      2. Change the required paths
      3. Run the code
      4. What is noteworthy here is that, if your system is crashed by the number of iterations, do the iterations in batches.
# 4. For the utilities
      After you have achieved your desired privacy results, you can now evaluate the anonymized datasets for utilities, follow these steps
      1. First you need to save the anonymized datasets according to the utility, i.e. race, age, or gender, use the copy new ipynb file to do this use the Copy New file in the For Utilities folder
      2. Change the paths, and use the respective notebook for each utility
      3. Make sure that all paths are updated
      
      

