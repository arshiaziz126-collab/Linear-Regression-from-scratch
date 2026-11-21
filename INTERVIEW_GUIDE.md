# GlycoCast: how to explain it in an interview (simple English)

## 30-second pitch
"I built linear regression from scratch in NumPy. I derived the gradients by hand and trained the model
with gradient descent, and I also solved it exactly with the normal equation. Both gave the same weights,
and they match scikit-learn. The model predicts how much a diabetes patient's disease will progress in one year
from 10 health measurements. It gets a cross-validated R² of about 0.5. I deployed it on Vercel with a FastAPI backend and a website that explains
every prediction, and a playground that shows how the learning rate and feature scaling affect gradient descent."

## Data in one line
"The Diabetes dataset from Efron et al. 2004: 442 patients, 10 baseline features (age, sex, BMI, blood pressure,
6 blood tests), and a target that measures disease progression one year later."

## Likely questions and simple answers

**What is linear regression?**
It finds the straight line (or flat surface) that best fits the data: `prediction = w·x + b`.
"Best" means the average squared error is as small as possible.

**What is gradient descent?**
Start with w = 0 and b = 0. Calculate which direction makes the error go down (the gradient) and take a small step that way.
Repeat until the error stops falling. It's like walking downhill in fog, one step at a time.

**What is the learning rate?**
The size of each step. Too small is very slow. Too big and you jump over the bottom, so the error explodes.
I show both cases in my playground.

**Why do two methods (gradient descent and the normal equation)?**
The normal equation gives the exact answer in one step. When gradient descent reaches the same answer, it proves my
gradients are correct. The normal equation becomes slow when there are very many features, and that's where gradient descent is better.

**Why did you scale the features?**
BMI is around 25 but cholesterol is around 190. Without scaling, the error surface becomes a narrow valley and
gradient descent bounces sideways and crawls. My chart shows this: after 300 steps it is still far from the answer.
With scaling it goes straight to the answer.

**What does R² mean?**
How much of the variation the model explains. 1 is perfect and 0 is no better than guessing the average.
Mine is about 0.4 on the test set and 0.5 with cross-validation.

**Isn't R² 0.4 low?**
For this dataset it's normal. Disease progression depends on many things that aren't in the data, like diet, medicine and genes.
scikit-learn gets exactly the same score. My model still reduces the error by 23% compared with guessing the average.

**Why is the test R² (0.385) different from the CV R² (0.507)?**
The test set has only 88 patients, so the score depends on which patients land in it.
Cross-validation averages 5 different splits, so it's more reliable. That's why I report both.

**Which features matter most?**
BMI, triglycerides, blood pressure and LDL push progression up.

**Your chart says total cholesterol lowers progression. Is that true?**
No. Total cholesterol and LDL are about 90% correlated, so the model splits the effect between them and
their weights partly cancel. This is called multicollinearity. It's a property of the model, not a medical fact.
Ridge regression (the L2 option in my code) reduces this.

**What is RMSE vs MAE?**
Both are average error. RMSE punishes big mistakes more because it squares them first. MAE is the plain average distance.

**What would you improve next?**
Use Ridge or Lasso to handle correlated features, try polynomial features for curved patterns,
and compare with tree-based models.

## Deployment questions

**How did you deploy it?**
"The backend is a FastAPI app in Python with three endpoints: `/api/info`, `/api/predict` and `/api/playground`.
The playground endpoint actually runs gradient descent live on the server for the settings the user picks.
The frontend is a small website that calls the API. Both are deployed together on Vercel as a serverless function,
so it doesn't go to sleep like Streamlit Cloud."

**Why not Streamlit?**
"Streamlit Community Cloud puts apps to sleep after 12 hours without visitors. With FastAPI on Vercel, the app is always
available, and the model (API) is separate from the user interface, which is how real ML products are built."

**What is an API?**
"A way for programs to talk to each other. The website sends the inputs as JSON, and the backend sends back the
prediction and the charts."

**How do you stop bad inputs?**
"FastAPI checks the request with Pydantic: steps must be between 10 and 1000, and the learning rate must be one of
the allowed values. Measurement values are clipped to the range seen in the data."
