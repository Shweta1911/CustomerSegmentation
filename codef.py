import streamlit as st
import sqlite3
import hashlib
import re
import streamlit as st
import base64
from heapq import nlargest
import os
from sklearn.linear_model import LogisticRegression
# Create a SQLite database and table
conn = sqlite3.connect('users.db')
cursor = conn.cursor()

conn.commit()
# Navigation options
email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'

# Function to validate the email
def validate_email(email):
    if re.match(email_pattern, email):
        return True
    else:
        return False
def is_strong_password(password):
    # Define a regular expression pattern for a strong password.
    pattern = "^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$"
    if re.match(pattern, password):
        return True
    else:
        return False
st.header("Main Page")
menu1=["Login","Register"]
choices1 = st.sidebar.selectbox("Select the option from the List",menu1)


if choices1 == 'Register':
        st.subheader("User Info Form")
    # name = st.text_input("Name")
        with st.form(key = 'user_info'):
            st.write('User Information')
    
            name = st.text_input(label="Name 📛")
            age = st.text_input(label="Lastname 📛")
            email = st.text_input(label="Email 📧")
            phone = st.text_input(label="Phone 📱")
            password = st.text_input(label="Password 📛", type="password")
            gender = st.radio("Gender 🧑", ("Male", "Female", "Prefer Not To Say"))
    
            submit_form = st.form_submit_button(label="Save Data", help="Click to register!", type="primary")
    
        # Checking if all the fields are non empty
            if submit_form:
                st.write("me")
               
    
                if name and age and email and phone and gender:
               
                    if not email or not password:
                        st.warning("Username and password are required.")
                    else:
                        hashed_password =password
                        if validate_email(email):
                            st.success(f"{email} is a valid email address.")
                            if is_strong_password(password):
                                st.success("Strong Password! ✔️")
                                try:

                                    cursor.execute("INSERT INTO users (firstname,lastname,username,password) VALUES (?, ?,?,?)", (name,age,email, hashed_password))
                                    conn.commit()
                                    st.success("Registration successful. You can now login.")
                                except sqlite3.IntegrityError:
                                    st.error("Username already exists. Please choose a different one.")
                            else:
                                st.warning("Weak Password! ❌")
                            

                        else:
                            st.error(f"{email} is not a valid email address.")
                else:
                    st.warning("Please fill all the fields")
if choices1 == 'Login':
       
        
        st.subheader("Login")
        username = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.sidebar.checkbox("Login"):
            hashed_password = password
            cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed_password))
            data = cursor.fetchone()
            if data:
                st.success("Logged in as {}".format(username))
                st.session_state.more_stuff = True
                st.success("Login successful!")
                menu=["Product ID Recommendation","Product Recommendataion","Customer Segmentation","Elbow method for segmentation"]
                choices = st.sidebar.selectbox("Select the option from the List",menu)


        

                if choices == '':
                        st.write('Select option from Dashboard')
                if choices=="Elbow method for segmentation":
                        import pandas as pd
                        from sklearn.cluster import KMeans
                        from sklearn.preprocessing import StandardScaler
                        import matplotlib.pyplot as plt

# Load your dataset
                        data = pd.read_excel('Online Retail.xlsx')  # Replace with the actual file path

# Extract relevant features for clustering
                        features = data[['Quantity', 'UnitPrice']]

# Standardize the features
                        scaler = StandardScaler()
                        features_scaled = scaler.fit_transform(features)

# Determine the optimal number of clusters using the elbow method
                        sse = []
                        for k in range(1, 11):
                                kmeans = KMeans(n_clusters=k, random_state=42)
                                kmeans.fit(features_scaled)
                                sse.append(kmeans.inertia_)

# Plot the elbow curve
                        plt.figure(figsize=(8, 6))
                        plt.plot(range(1, 11), sse, marker='o')
                        plt.title('Elbow Method for Optimal k')
                        plt.xlabel('Number of Clusters (k)')
                        plt.ylabel('Sum of Squared Distances (SSE)')
                        plt.grid(True)
                        
                        st.pyplot()

# Choose the optimal k based on the elbow method
                        optimal_k = 3  # Replace with your chosen value

# Perform k-means clustering with the optimal k
                        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
                        data['Cluster'] = kmeans.fit_predict(features_scaled)

# Display the results
                        st.write(data[['CustomerID', 'Quantity', 'UnitPrice', 'Cluster']])


                if choices == 'Customer Segmentation':
       
                        st.subheader("Customer Segmentation")       
                        import numpy as np
                        import pandas as pd
                        import matplotlib.pyplot as plt
                        import math
                        import matplotlib.mlab as mlab
                        import datetime
                        import scipy
                        import scipy.stats as stats
                        import seaborn as sns

                        import os



                        OR_df=pd.read_excel('Online Retail.xlsx')
                        OR_df.head()
                        OR_df.Country.value_counts().reset_index().head(20)
                        OR_df.CustomerID.unique().shape
                        (OR_df.CustomerID.value_counts()/sum(OR_df.CustomerID.value_counts())*100).head(13).cumsum()
                        OR_df.StockCode.unique().shape
                        OR_df.Description.unique().shape
                        des_df=OR_df.groupby(['Description','StockCode']).count().reset_index()
                        des_df.StockCode.value_counts()[des_df.StockCode.value_counts()>1].reset_index().head()
                        OR_df.Quantity.describe()
                        OR_df.UnitPrice.describe()
                        OR_df=OR_df[OR_df.Country=='United Kingdom']
                        st.write(OR_df.head())
                        OR_df['Amount']=OR_df.Quantity*OR_df.UnitPrice
                        OR_df['Amount'].head()
                        OR_df=OR_df[~(OR_df['Amount']<0)]
                        OR_df.head()
                        OR_df=OR_df[~(OR_df.CustomerID.isnull())]
                        st.write(OR_df.shape)
                        OR_df.head()
                        reference_date=OR_df.InvoiceDate.max()
                        reference_date=reference_date+datetime.timedelta(days=1)#timedelta function returns to total number of seconds
                        st.write(OR_df.InvoiceDate.max(),OR_df.InvoiceDate.min())
                        reference_date
                        OR_df['days_since_last_purchase']=reference_date-OR_df.InvoiceDate
                        OR_df['days_since_last_purchase_num'] = OR_df['days_since_last_purchase'].dt.days
                        OR_df['days_since_last_purchase_num'].head()  
                        customer_history_df = OR_df.groupby('CustomerID')['days_since_last_purchase_num'].min().reset_index()
                        customer_history_df.rename(columns={'days_since_last_purchase_num':'Recency'}, inplace=True)
                        st.write(customer_history_df.describe())
                        customer_history_df.head()
                        x=customer_history_df.Recency
                        mu=np.mean(x)
                        sigma=math.sqrt(np.var(x))
                        n,bins,patches=plt.hist(x,1000,facecolor='blue',alpha=0.75)
                        y=scipy.stats.norm.pdf(bins,mu,sigma)
                        l=plt.plot(bins,y,'r--',lw=2)
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        plt.xlabel('Recency in days')
                        plt.ylabel('Number of transactions')
                        plt.title('Histogram of Sales Recency')
                        plt.grid(True)
       
                        st.pyplot()


                        plt.savefig("sine_wave_plot.png")
                        customer_monetary_val=OR_df[['CustomerID','Amount']].groupby('CustomerID').sum().reset_index()
                        customer_history_df=customer_history_df.merge(customer_monetary_val,how='outer')
                        customer_history_df.Amount=customer_history_df.Amount+0.001
                        customer_freq=OR_df[['CustomerID','Amount']].groupby('CustomerID').count().reset_index()
                        customer_freq.rename(columns={'Amount':'Frequency'},inplace=True)
                        customer_history_df=customer_history_df.merge(customer_freq,how='outer')

                        customer_history_df=pd.DataFrame(customer_history_df,columns=['CustomerID','Recency','Amount','Frequency'])
                        customer_history_df.head()
                        from sklearn import preprocessing
                        customer_history_df['Recency_log'] = customer_history_df['Recency'].apply(math.log)
                        customer_history_df['Frequency_log'] = customer_history_df['Frequency'].apply(math.log)
                        customer_history_df['Amount_log'] = customer_history_df['Amount'].apply(math.log)
                        feature_vector=['Recency_log','Frequency_log','Amount_log']
                        X=customer_history_df[feature_vector].values
                        scaler=preprocessing.StandardScaler()
                        X_scaled=scaler.fit_transform(X)
        
                        plt.scatter(customer_history_df.Recency_log,customer_history_df.Amount_log,alpha=0.5)

                        plt.scatter(customer_history_df.Frequency_log,customer_history_df.Amount_log,alpha=0.5)



                        x=customer_history_df.Amount_log
                        n,bins,patches=plt.hist(x,1000,facecolor='b',alpha=0.8)
                        plt.xlabel('Log of Sales Amount')
                        plt.ylabel('Probability')
                        plt.title('Histogram of log transformed monetary value ')
                        plt.grid(True)
                        st.pyplot()
                        plt.savefig("a.png")
                        from mpl_toolkits.mplot3d import Axes3D
                        fig=plt.figure(figsize=(10,8))
                        ax=fig.add_subplot(111,projection='3d')
                        xs=customer_history_df.Recency_log
                        ys=customer_history_df.Frequency_log
                        zs=customer_history_df.Amount_log
                        ax.scatter(xs,ys,zs,s=5)
                        ax.set_xlabel('Recency')
                        ax.set_ylabel('Frequency')
                        ax.set_zlabel('Monetary value')
                        st.pyplot()


                        plt.savefig("b.png")
                        from sklearn.cluster import KMeans
                        import matplotlib.cm as cm
                        from sklearn.metrics import silhouette_samples,silhouette_score
                        X=X_scaled
                        cluster_centers=dict()
                        for n_clusters in range(3,6,2):
                                fig,(ax1,ax2)=plt.subplots(1,2)
                                fig.set_size_inches(18,7)
                                ax1.set_xlim([-0.1,1])
                                ax1.set_ylim([0,len(X)+(n_clusters+1)*10])
    
                                clusterer=KMeans(n_clusters=n_clusters,random_state=10)
                                cluster_labels=clusterer.fit_predict(X)
    
                        silhouette_avg=silhouette_score(X,cluster_labels)
                        cluster_centers.update({n_clusters:{'cluster_centre':clusterer.cluster_centers_,
                                       'silhouette_score':silhouette_avg,
                                       'labels':cluster_labels}
                           })
    
                        sample_silhouette_values=silhouette_samples(X,cluster_labels)
                        y_lower=10
                        for i in range(n_clusters):
                                ith_cluster_silhouette_values=sample_silhouette_values[cluster_labels==i]
                                ith_cluster_silhouette_values.sort()
                                size_cluster_i=ith_cluster_silhouette_values.shape[0]
                                y_upper=y_lower+size_cluster_i
    
                                cmap = cm.get_cmap("Spectral")
                                color=cmap(float(i)/n_clusters)
                                ax1.fill_betweenx(np.arange(y_lower,y_upper),0,
                                ith_cluster_silhouette_values,facecolor=color,edgecolor=color,alpha=0.75)
                                ax1.text(-0.05,y_lower+0.5*size_cluster_i,str(i))
                                y_lower=y_upper+10 # 10 for 0 samples
        
                        ax1.set_title('The silhouette plot for the various clusters')
                        ax1.set_xlabel('The silhouette coefficient values')
                        ax1.set_ylabel('Cluster_label')
                        ax1.axvline(x=silhouette_avg,color='red',linestyle='--')
                        ax1.set_yticks([])
                        ax1.set_xticks([-0.1,0,0.2,0.4,0.6,0.8,1])
    
                        colors=cmap(cluster_labels.astype(float)/n_clusters)
                        feature1=0
                        feature2=2
                        ax2.scatter(X[:,feature1],X[:,feature2],marker='.',s=30,
                                lw=0,alpha=0.7,edgecolor='k',c=colors)
                        centers=clusterer.cluster_centers_
                        ax2.scatter(centers[:,feature1],centers[:,feature2],marker="o",
                                alpha=1,c='white',s=200,edgecolor='k')

                        for i,c in enumerate(centers):
                                ax2.scatter(c[feature1],c[feature2],marker='$%d$'%i,alpha=1,
                                edgecolor='k',s=50)
                        ax2.set_title('The visulization of clustered data')
                        ax2.set_xlabel('Feature space for the 2nd feature(Monetary Value)')
                        ax2.set_ylabel('Feature space for the 1st feature(Recency)')
                        plt.suptitle('Silhouetee analysis for KMeans clustering on sample data' 'with n_clusters=%d'
                         % n_clusters,fontsize=14,fontweight='bold')
                        st.pyplot()


                        plt.savefig("c.png")
                        from sklearn.cluster import KMeans
                        from sklearn.metrics import silhouette_samples, silhouette_score
                        import matplotlib.cm as cm

                        X = X_scaled

                        cluster_centers = dict()
                        for n_clusters in range(3,6,2):
                                fig, (ax1, ax2) = plt.subplots(1, 2)
    #ax2 = plt.subplot(111, projection='3d')
                                fig.set_size_inches(18, 7)
                                ax1.set_xlim([-0.1, 1])
                                ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

                                clusterer = KMeans(n_clusters=n_clusters, random_state=10)
                                cluster_labels = clusterer.fit_predict(X)

                                silhouette_avg = silhouette_score(X, cluster_labels)
                                cluster_centers.update({n_clusters :{
                                        'cluster_center':clusterer.cluster_centers_,
                                        'silhouette_score':silhouette_avg,
                                        'labels':cluster_labels}
                                })
                        sample_silhouette_values = silhouette_samples(X, cluster_labels)
                        y_lower = 10
                        for i in range(n_clusters):
                                ith_cluster_silhouette_values = \
                                sample_silhouette_values[cluster_labels == i]

                                ith_cluster_silhouette_values.sort()

                                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                                y_upper = y_lower + size_cluster_i

                                cmap=cm.get_cmap('Spectral')
                                color = cmap(float(i) / n_clusters)
                                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                        0, ith_cluster_silhouette_values,
                                        facecolor=color, edgecolor=color, alpha=0.7)

                        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                        y_lower = y_upper + 10  # 10 for the 0 samples
                        ax1.set_title("The silhouette plot for the various clusters.")
                        ax1.set_xlabel("The silhouette coefficient values")
                        ax1.set_ylabel("Cluster label")
                        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
                        ax1.set_yticks([])
                        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
                        colors = cmap(cluster_labels.astype(float) / n_clusters)
                        feature1 = 1
                        feature2 = 2
                        ax2.scatter(X[:, feature1], X[:, feature2], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')
    
                        centers = clusterer.cluster_centers_
                        ax2.scatter(centers[:, feature1], centers[:, feature2], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')
                        for i, c in enumerate(centers):
                                ax2.scatter(c[feature1], c[feature2], marker='$%d$' % i, alpha=1,
                                s=50, edgecolor='k')
                        ax2.set_title("The visualization of the clustered data.")
                        ax2.set_xlabel("Feature space for the 2nd feature (Monetary Value)")
                        ax2.set_ylabel("Feature space for the 1st feature (Frequency)")
                        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                                 "with n_clusters = %d" % n_clusters),
                                fontsize=14, fontweight='bold')
                        st.pyplot()
                        plt.savefig("d.png")
                        for i in range(3,6,2):
                                print('for {} number of clusters'.format(i))
                                cent_transformed=scaler.inverse_transform(cluster_centers[i]['cluster_center'])
                                st.write(pd.DataFrame(np.exp(cent_transformed),columns=feature_vector))
                                st.write('Silhouette score for cluster {} is {}'.format(i,cluster_centers[i]['silhouette_score']))
                        
                        labels=cluster_centers[5]['labels']
                        customer_history_df['num_cluster5_labels']=labels
                        labels=cluster_centers[3]['labels']
                        customer_history_df['num_cluster3_labels']=labels

                        customer_history_df.head()
                        import plotly as py
                        import plotly.graph_objs as go
                        py.offline.init_notebook_mode()

                        x_data=['Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5']
                        cutoff_quantile=100
                        field_to_plot='Recency'
                        y0 = customer_history_df[customer_history_df['num_cluster5_labels']==0][field_to_plot].values
                        y0 = y0[y0<np.percentile(y0, cutoff_quantile)]
                        y1=customer_history_df[customer_history_df['num_cluster5_labels']==1][field_to_plot].values
                        y1=y1[y1<np.percentile(y1,cutoff_quantile)]
                        y2 = customer_history_df[customer_history_df['num_cluster5_labels']==2][field_to_plot].values
                        y2 = y2[y2<np.percentile(y2, cutoff_quantile)]
                        y3 = customer_history_df[customer_history_df['num_cluster5_labels']==3][field_to_plot].values
                        y3 = y3[y3<np.percentile(y3, cutoff_quantile)]
                        y4 = customer_history_df[customer_history_df['num_cluster5_labels']==4][field_to_plot].values
                        y4 = y4[y4<np.percentile(y4, cutoff_quantile)]
                        y_data=[y0,y1,y2,y3,y4]
                        colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)', 'rgba(255, 65, 54, 0.5)',
                                'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)']
                        traces=[]

                        for xd,yd,cls in zip(x_data,y_data,colors):
                                traces.append(go.Box(y=yd,
                                name=xd,
                                boxpoints=False,
                                jitter=0.5,
                                whiskerwidth=0.2,
                                fillcolor=cls,
                                marker=dict(size=2,),
                                line=dict(width=1),
                                ))
                        layout=go.Layout(
                        title='Difference in sales {} from cluster to cluster'.format(field_to_plot),
                        yaxis=dict(autorange=True,
                        showgrid=True,
                        zeroline=True,
                        dtick=50,
                        gridcolor='rgb(255, 255, 255)',
                        gridwidth=0.1,
                        zerolinecolor='rgb(255,255,255)',
                        zerolinewidth=2,),
                        margin=dict(
                                l=40,
                                r=30,
                                b=80,
                                t=100,
                        ),
                        paper_bgcolor='rgb(243, 243, 243)',
                        plot_bgcolor='rgb(243, 243, 243)',
                        showlegend=False
                        )
                        fig=go.Figure(data=traces,layout=layout)
                        py.offline.iplot(fig)

                if choices=="Product ID Recommendation":
                        
                                import streamlit as st
                                import pandas as pd
                                from surprise import Dataset, Reader, SVD
                                from surprise.model_selection import train_test_split
                                from surprise import accuracy

# Load the dataset
                                df = pd.read_csv('OnlineRetail.csv')

# Convert implicit feedback to binary ratings (1 for interaction, 0 for no interaction)
                                df['Rating'] = (df['Quantity'] > 0).astype(int)

# Streamlit UI
                                st.title("Product ID Recommendation")
                                user_id = st.text_input(label="Customer ID")

                                if not user_id:
                                        st.warning("Input box is empty. Please enter something.")
                                else:
    # Create a Surprise reader and load the dataset
                                        reader = Reader(rating_scale=(0, 1))
                                        data = Dataset.load_from_df(df[['CustomerID', 'Description', 'Rating']], reader)

    # Build the collaborative filtering model (SVD)
                                        trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
                                        model = SVD()
                                        model.fit(trainset)

    # Get items not interacted by the user
                                        items_not_rated = df.loc[~df['Description'].isin(df[df['CustomerID'] == int(user_id)]['Description']), 'Description'].unique()

    # Get predicted ratings for the items not rated by the user
                                        item_ratings = [(item_id, model.predict(int(user_id), item_id).est) for item_id in items_not_rated]

    # Sort the items by predicted ratings in descending order
                                        item_ratings = sorted(item_ratings, key=lambda x: x[1], reverse=True)

    # Display top N recommendations
                                        top_n = 5
                                        st.write(f"Top {top_n} Recommendations for User Using SVD {user_id}:")
                                        for i, (item_name, est_rating) in enumerate(item_ratings[:top_n], 1):
                                                st.write(f"{i}. Item {item_name}: Estimated Rating = {est_rating}")

                                        reader = Reader(rating_scale=(0, 1))
                                        data = Dataset.load_from_df(df[['CustomerID', 'Description', 'Rating']], reader)

                                        #collaborative filtering model (FunkSVD)
                                        trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
                                        model = SVD()
                                        model.fit(trainset)

    # Get items not interacted by the user
                                        items_not_rated = df.loc[~df['Description'].isin(df[df['CustomerID'] == int(user_id)]['Description']), 'Description'].unique()

    # Get predicted ratings for the items not rated by the user
                                        item_ratings = [(item_id, model.predict(int(user_id), item_id).est) for item_id in items_not_rated]

    # Sort the items by predicted ratings in descending order
                                        item_ratings = sorted(item_ratings, key=lambda x: x[1], reverse=True)

    # Display top N recommendations
                                        top_n = 5
                                        st.write(f"Top {top_n} Recommendations for User Using collaborative filtering model {user_id}:")
                                        for i, (item_name, est_rating) in enumerate(item_ratings[:top_n], 1):
                                                 st.write(f"{i}. Item {item_name}: Estimated Rating = {est_rating}")





                if choices == 'Product Recommendataion':
                        st.subheader("Product Recommendation")
                        Algorithm = st.radio("Algorithm", ("Collaborative filtering", "Singular Value Decomposition (SVD)"))
                        st.write("Selected Algorithm is "+Algorithm)
                        if Algorithm=="Collaborative filtering":
                                st.write(" I am in Collaborative filtering code")
                
                                import pandas as pd
                                df = pd.read_excel('Online Retail.xlsx')
                                st.write(df.head())
                                df = df.loc[df['Quantity'] > 0]
                                df.info()
                                df['CustomerID'].isna().sum()
                                df = df.dropna(subset=['CustomerID'])
                                customer_item_matrix = df.pivot_table(
                                        index='CustomerID',
                                        columns='StockCode',
                                        values='Quantity',
                                        aggfunc='sum'
                                )
                                customer_item_matrix.loc[12481:].head()
                                st.write(customer_item_matrix.shape)
                                customer_item_matrix = customer_item_matrix.applymap(lambda x: 1 if x > 0 else 0)
                                from sklearn.metrics.pairwise import cosine_similarity
                                user_user_sim_matrix = pd.DataFrame(cosine_similarity(customer_item_matrix))
                                user_user_sim_matrix.head()
                                user_user_sim_matrix.columns = customer_item_matrix.index

                                user_user_sim_matrix['CustomerID'] = customer_item_matrix.index
                                user_user_sim_matrix = user_user_sim_matrix.set_index('CustomerID')
                                user_user_sim_matrix.head()
                                user_user_sim_matrix.loc[12350.0].sort_values(ascending=False).head(10)
                                user_user_sim_matrix.loc[12350.0].sort_values(ascending=False)
                                items_bought_by_A = customer_item_matrix.loc[12350.0][customer_item_matrix.loc[12350.0]>0]
                                st.write("Items Bought by A: ")
                                st.write(items_bought_by_A)
                                items_bought_by_B = customer_item_matrix.loc[17935.0][customer_item_matrix.loc[17935.0]>0]
                                st.write("Items bought by B:")
                                st.write(items_bought_by_B)

                                print()
                                items_to_recommend_to_B = set(items_bought_by_A.index) - set(items_bought_by_B.index)
                                st.write("Items to Recommend to B ")
                                st.write(items_to_recommend_to_B)
                                df.loc[df['StockCode'].isin(items_to_recommend_to_B),['StockCode', 'Description']].drop_duplicates().set_index('StockCode')
                                item_item_sim_matrix = pd.DataFrame(cosine_similarity(customer_item_matrix.T))
                                item_item_sim_matrix.columns = customer_item_matrix.T.index

                                item_item_sim_matrix['StockCode'] = customer_item_matrix.T.index
                                item_item_sim_matrix = item_item_sim_matrix.set_index('StockCode')
                                st.write(item_item_sim_matrix)
                                top_10_similar_items = list(item_item_sim_matrix.loc[23166].sort_values(ascending=False).iloc[:10].index)

                                st.write(top_10_similar_items)
                                st.write()
                                st.write(df.loc[
                                df['StockCode'].isin(top_10_similar_items),
                                ['StockCode', 'Description']
                                ].drop_duplicates().set_index('StockCode').loc[top_10_similar_items])
                                user_id = 12350.0  # Replace with the user ID of interest
                                actual_liked_items = customer_item_matrix.loc[user_id][customer_item_matrix.loc[user_id] > 0].index

# Assuming you have the recommended items for the user
                                recommended_items = df.loc[df['StockCode'].isin(items_to_recommend_to_B), 'StockCode'].unique()

# Calculate precision
                                precision = len(set(recommended_items) & set(actual_liked_items)) / len(recommended_items) if len(recommended_items) > 0 else 0.0

# Display precision
                                st.write(f"Precision for user {user_id}: {precision}")
                        else:
                                user_id = 12583
                                st.write("I am on SVD")
                                
                                import pandas as pd
                                from surprise import Dataset, Reader, SVD
                                from surprise.model_selection import train_test_split
                                from surprise import accuracy
                                df = pd.read_csv('OnlineRetail.csv')
                                df['Rating'] = (df['Quantity'] > 0).astype(int)
                                reader = Reader(rating_scale=(0, 1))
                                data = Dataset.load_from_df(df[['CustomerID', 'StockCode', 'Rating']], reader)

                                trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
                                model = SVD()
                                model.fit(trainset)
                                predictions = model.test(testset)
                                

                                st.write(accuracy.rmse(predictions))
                                
                                
                                user_items = df[df['CustomerID'] == user_id]['StockCode'].unique()


                                user_recommendations = [(item_id, model.predict(user_id, item_id).est) for item_id in user_items]


                                user_recommendations = sorted(user_recommendations, key=lambda x: x[1], reverse=True)


                                top_n = 5
                                st.write(f"Top {top_n} Recommendations for User {user_id}:")
                                for i, (item_id, est_rating) in enumerate(user_recommendations[:top_n], 1):
                                                st.write(f"{i}. Item {item_id}: Estimated Rating = {est_rating}")
                                from sklearn.metrics import confusion_matrix, accuracy_score


                                threshold = 0.5  # You may need to adjust this threshold based on your problem
                                predicted_ratings = [1 if pred > threshold else 0 for uid, iid, true_r, pred, _ in predictions]
                                true_ratings = [int(true_r) for uid, iid, true_r, pred, _ in predictions]

# Calculate confusion matrix
                                conf_matrix = confusion_matrix(true_ratings, predicted_ratings)
                                from sklearn.metrics import classification_report
                                print("Confusion Matrix:\n", conf_matrix)

# Calculate accuracy
                                accuracy = accuracy_score(true_ratings, predicted_ratings)
                                print("Accuracy:", accuracy)
                                class_report = classification_report(true_ratings, predicted_ratings)

# Display the classification report
                                print("Classification Report:\n", class_report)

                                st.write(class_report)
                                import matplotlib.pyplot as plt
                                from sklearn.metrics import classification_report
                                from sklearn.metrics import confusion_matrix
                                import seaborn as sns


                                threshold = 0.5  # You may need to adjust this threshold based on your problem
                                predicted_ratings = [1 if pred > threshold else 0 for uid, iid, true_r, pred, _ in predictions]
                                true_ratings = [int(true_r) for uid, iid, true_r, pred, _ in predictions]


                                class_report = classification_report(true_ratings, predicted_ratings, output_dict=True)


                                precision = [class_report[str(i)]['precision'] for i in range(2)]
                                recall = [class_report[str(i)]['recall'] for i in range(2)]


                                labels = ['Class 0', 'Class 1']
                                fig, ax = plt.subplots()
                                ax.pie(precision, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightskyblue'])
                                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                                ax.set_title('Precision Pie Chart')
                                plt.grid(True)
       
                                st.pyplot()


                                fig, ax = plt.subplots()
                                ax.pie(recall, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightskyblue'])
                                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                                ax.set_title('Recall Pie Chart')
                                plt.grid(True)
       
                                st.pyplot()
                        






  
       
                
                
        

      
         
       

       