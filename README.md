# Large-Scale Quantatative Analysis of Judging in Professional Boxing

This repo serves to analyze a database of 1,000+ of boxing matches with accompaning round-by-round scorecards. Though our analysis we determine the relative important of different aspects of a fighter's performance. We also evalute top judges and analyze differences in their style that emerge from the data. Lastly, we create an automated judging system that is interpretable and able to accurately predict judging outcomes.

This code was developed built as part of research collaboration between Jabbr and INS Quebec.

The stats used in the investigation are entirely generasted using [Jabbr's DeepStrike platform](https://jabbr.ai/blog/deepstrike-stats-explained).

NOTE: Statistics were generated using an outdated version of DeepStrike. The version of DeepStrike used to generate the statistics for this study was trained on 194 bouts, featuring 171,301 punches thrown. All bouts were fully annotated frame by frame from start to end. The newest version of DeepStrike (as of September 2025) is trained on nearly twice as much data, 356 fights, featuring a total of 332,860 punches thrown. Furthermore, all statistics were generated using the "dirty feed", which was uploaded to video streaming sites in either 720p or 1080p resolution. Dirty feeds only present a single view of the action at a given time and thus, are prone to occulsion issues. For optimial tracking accuracy a 4-camera system should be used so that DeepStrike can have miltiple perspectives on the action taking place.

Excel statsheets generated using DeepStrike can be found here:

https://drive.google.com/drive/folders/1wPaEDpTRJ_LDexz60pMsLhxPB9KFpohu?usp=drive_link


# Data Analysis
Once videos are downloaded they are manually uploaded to [Jabbr's DeepStrike platform](https://jabbr.ai/deepstrike). The resulting excel stat sheets are placed in a folder called "stats" in the analysis folder. This stat data is read into an analysis script along with the scoring data. Since DeepStrike currently lacks the ability to indentify knockdowns or deductions, any round that isn't scored 10-9 or 9-10 is excluded from the dataset. Various major statistics are normalized and graphed as scatter plots and histograms in relation to how many judges scored for that particular fighter.

After exploring a number of techiques to correlate stats to judging outcomes including Linear and ordered logisticc regressions, two approaches/models were primarily used, gradient descent and a MLP neural network. Gradient descent is used to indicate the relative impact of different aspects of a fighter's performance. Gradient descent is also used on data for specific judges, allowing me to investigate how judges differ in the importance they put on different aspects of fighters' performances. For gradient descent a formula is used comprising a number of +/- stats each multiplied by a coefficent variable dynamically set by the program. The product of these +/- stats and their coffiecents are added together and their sum is made to fit between -1 and 1 using a sigmoid activation function. All stats used in gradient descent are +/- values for the particular fighter being considered. A preidcted value of -1 represents a predicted loss while 1 represents a predicted win. Many different combinations of parameters are explored. Below is an example of a formula with few parameters that results in good predictive performance. Note that aggression power is a compound metric, between 0 and 100 given to each fighter, that measures throwing punches with high power-commit. Similarly, positional pressure is a compound metric that measures when you have your opponent near the ropes or corners. Also, it is important to note that the various impact parameters, only count landed punches that fall into that particular impact category. Punches that are blocked, landing on an opponents guard are not counted here, along with punches that miss the opponent entirely.

prediction$`=a(`$min impact +/-$`)+b(`$low impact +/-$`)+c(`$mid impact +/-$`)+d(`$high impact +/-$`)+e(`$max impact +/-$`)+f(`$positional pressure +/-$`)+g(`$agression power +/-$`)+h(`$accuracy +/-$`)`$

Here are the coefficent values for each stat, after gradient descent is run, fitting this predictive model to a dataset of about 5000 rounds.

You can clearly the exponential relationship between the different impact statistics, with a punch in each impact category given nearly 2x the value of a punch in the impact category below it.

    min              : 0.0311
    low              : 0.0578
    mid              : 0.1299
    high             : 0.2215
    max              : 0.4203
    accuracy         : 0.0160
    pressurePosition : 0.0109
    aggressionPower  : 0.0920

And here are the accuracy and cost metrics associate with the above coefficient values.

    Accuracy for Unanimous Rounds:  84.756%
    Accuracy for Split Rounds:      57.955%
    Total Accuracy:                 77.271%
    Average Cost:                   0.40495
    Median Cost:                    0.16931


TensorFlow Keras is used to implement an MLP neural network. Unlike gradient descent this apprach enables us to model non-linear relationships between stats and judge scores. However, fewer parameters must be used to avoid overfitting and needs for greater amount of data means that we are not able to model a subsection of data, specific to a particular judge without en countering overfitting issues. However, this approach allows us to model more complicated realtionships between parameters and result in marginally better predictive accuracy.

Flags available when running analysis.py:

    -mlp                         Creates a multi-layer perception to model data instead of using gradient descent.
    -best                        Don't run gradient descent, instead use presaved coeffient values (params for saved values must match parameters array specified in main)
    -j "<Judge Name>"            Only data for the specified judge is used in modeling & analysis.
    -split <decimal 0-1>         Creates a testing data set of specifed ratio (ratio also applied to creation of validation set if used w -mlp)
    -dt <decimal 0-1>            Filters out close rounds when ranking judges by setting a "disgreement threshold". All rounds with an absolute predicted value < the threshold are ignored.
    -costrank                    Rank judges by the average cost of their scores, instead of by their % accuracy.
    -samplerank                  When ranking judges randomly some of their number of rounds and ignore the rest so that all judges' in the ranking have the same sample size.
    -combo <+int>                Runs gradient descent with every combination of <int> params that can be made from the array of parameters specified in main, prints best combinations.
    -combostart <+int>           Like -combo but will always use startingParms, finding the best <int> parameters to add to startingParams.
    -lookup                      After all analysis has been done, prompting the user to enter the names and a date of a fight and print out round by round details for that bout.
    -includeinserted             Add to the dataset fights whose stat sheets are marked as INCLUDED


### Parameter Cycling
Using the -combo and -combostart flags you automatically run gradient descent or mlp multiple times, each time changing the combination of parameters used. A fixed number of parameter "slots" are maintained even has the parameters themselves are swapped out. Once this process is complete the system prints the sets of parameters that resulted in the lowest cost. This process can take a very long time, depending on the number of parameter "slot", specified in the command line, and the total number of parameters to be considered. Parameters that are critical for solid predicitve accuracy likely measure something important to judges' considerations.


### Correlation Cooefficients
We can run a number of metrics to quantify just how correlated our parameters are to judging outcomes. Here we calculate the Pearson and Spearman correlation coefficent for each parameter and rank them by how strongly correlated they are to judges' scores. In our analysis we see the parameters with the highest correlation values go to the number of punches landed and agresssionPower, a measures of throwing punches with high power-commit. Alongside the parameters, we also calculate coorelation coefficients for our predicted scores. This value is listed under "heuristic" and has a considerably greater Pearson and Spearman score than any single parameter, implying it to be the greatest single incidator of judge score at our disposal.


### Correlations Matrix
Yet to be implemented.


### Laterality
These is a lot of discussion in the world of boxing, theorizing that southpaw boxers may have a competitive advantage because due to the fact that they are encountered less frequently. As part of our dataset we have access to round-by-round stance data, measuring what percentage of a round each boxer spends in a orthodox, southpaw or squared stance. Looking only at 10-9 and 9-10 rounds we are able to calculate the win-rate for southpaw boxers compared to orthodox boxers.


### Scatter Plots
Scatter plots are made for every parameter, plotting every single round performance, given by a boxer. This results in there being two points plotted for every round, one for the winner and one for the loser. The x axis indicates how the boxer scored on that particular metric relative to their opponent (ie. their +/-). The y axis shows how many of the three judges score the round in favor of the boxer in question.

These graphs are useful for viewing small datasets but with thousands of rounds, the points overlap and prevent us from getting a sense of the actual distribution.

There will be two different sets of scatter plots generated, normalized and non-normalized graphs. Normalized graphs will all have the same bounds and tick marks, with the x axis representing each point's distance from the mean, 0, in terms of standard deviations.

### Histograms


### Rankings Rounds


### Ranking Fights


### Ranking Judges


### Analysis of Judge Groupings
Grouping based on nationality and years of experience
Yet to be implemented.
