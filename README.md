## Dataset

We have structured the dataset directory in advance.  
You can download the complete dataset package from the following link:

🔗 https://drive.google.com/file/d/1o9h3Kh6zovW4FIHpNBGnYIRSbGCu-qPt/view

After downloading, simply place the corresponding data into the predefined directories — no additional organization is needed.

To simplify the process, we have already included all necessary files for **HR-UBnormal** within the dataset package.

## Testing

We have provided the pre-trained weights within the downloaded package.  
Once the dataset is downloaded and the required environment is set up, you can directly run the evaluation using the following command:

```bash
python eval.py --dataset [ShanghaiTech, ShanghaiTech-HR, UBnormal, UBnormal-HR]
