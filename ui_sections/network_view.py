# ui_sections/network_view.py
import streamlit as st
import networkx as nx
import plotly.graph_objects as go


def render_network_analysis(topic_network_graph, num_topics_config, topic_labels_config, df_processed): 
    st.header("🕸️ Topic Co-occurrence Network")
    st.info("""
    **What is Topic Co-occurrence Network Analysis?**
    This technique visualizes the relationships between different topics. If two topics frequently appear together within the same set of documents (in our case, customer reviews), they are considered to have a strong co-occurrence.
    The network graph helps us see these connections visually.

    **How it's used in this project:**
    * **Nodes (Circles):** Represent the `[NUM_TOPICS]` topics discovered by LDA.
    * **Edges (Lines):** Connect topics that are discussed together in the same reviews. The thickness or darkness of a line indicates how frequently those two topics co-occur.
    * **Node Size/Color:** Can represent topic prevalence or centrality (importance within the network).

    **Business Impact:**
    * **Understand Compound Issues:** Identify if certain problems or praises are linked (e.g., "Sizing issues" often discussed with "Return Process").
    * **Holistic Customer View:** See how different aspects of the customer experience are connected in the customer's mind.
    * **Prioritize Efforts:** Addressing a central topic in the network might have a cascading positive effect on related topics.
    * **Discover Unexpected Relationships:** Uncover connections between themes that might not be immediately obvious.
    """)
    st.markdown("---") 
    
    st.markdown("""
    This interactive network visualizes which topics are frequently discussed *together within the same reviews*. 
    Hover over nodes (topics) for details and edges (lines) for co-occurrence strength. You can pan, zoom, and explore connections.
    - **Nodes (Circles):** Represent the topics. Larger nodes indicate topics that are more prevalent overall. Node color intensity reflects centrality.
    - **Edges (Lines):** Connect topics that co-occur. Thicker lines signify stronger co-occurrence.
    """)
    if topic_network_graph is not None and topic_network_graph.number_of_nodes() > 0 :
        if not nx.is_empty(topic_network_graph):
            pos = nx.spring_layout(topic_network_graph, k=0.7, iterations=50, seed=42)
            
            edge_x, edge_y, edge_hover_texts = [], [], []
            for edge in topic_network_graph.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_hover_texts.extend([f"Co-occurrence: {edge[2].get('weight', 1)}", f"Co-occurrence: {edge[2].get('weight', 1)}", ""])

            edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), 
                                     hoverinfo='text', text=edge_hover_texts, mode='lines')

            node_x_vals, node_y_vals, node_hover_texts_list, node_sizes_viz, node_color_vals, node_text_labels = [], [], [], [], [], []
            centrality = nx.degree_centrality(topic_network_graph) if topic_network_graph.number_of_nodes() > 1 else {node_id: 0 for node_id in topic_network_graph.nodes()}
            max_centrality = max(centrality.values()) if centrality and any(centrality.values()) else 1.0
            
            
            topic_prevalence_map = df_processed['dominant_lda_topic'].value_counts().to_dict() if df_processed is not None else {}


            for node_id in topic_network_graph.nodes():
                x, y = pos[node_id]
                node_x_vals.append(x); node_y_vals.append(y)
                label = topic_labels_config.get(int(node_id), f"Topic {node_id}")
                node_text_labels.append(label)
                hover_text = f"<b>{label}</b><br>Centrality: {centrality.get(node_id, 0):.3f}"
                
                # Use prevalence for node size, with a fallback
                base_size_for_node = topic_prevalence_map.get(int(node_id), 1) # Get prevalence count
                scaled_node_size = 10 + (base_size_for_node / (max(topic_prevalence_map.values()) if topic_prevalence_map else 1) * 40) if topic_prevalence_map else 20
                node_sizes_viz.append(scaled_node_size)

                node_color_vals.append(centrality.get(node_id, 0))
                node_hover_texts_list.append(hover_text)
            
            node_trace = go.Scatter(x=node_x_vals, y=node_y_vals, mode='markers+text', 
                                     hoverinfo='text', text=node_hover_texts_list, 
                                     textfont=dict(size=10, color='#1A5276'), textposition='top center',
                                     customdata=node_text_labels, 
                                     marker=dict(showscale=True, colorscale='Blues', reversescale=False, color=node_color_vals, 
                                                 size=node_sizes_viz, line_width=2, line_color='black',
                                                 colorbar=dict(thickness=15, title=dict(text='Node Centrality', side='right'), xanchor='left')))
            node_trace.text = node_text_labels

            fig_network = go.Figure(data=[edge_trace, node_trace],
                                         layout=go.Layout(title=dict(text='<b>Interactive Topic Co-occurrence Network</b>', x=0.5, font_size=20),
                                                          showlegend=False, hovermode='closest', height=800,
                                                          margin=dict(b=20,l=5,r=5,t=50),
                                                          annotations=[dict(text="Hover: Topic details. Darker/Larger nodes: More central/prevalent. Lines: Co-occurrence.", showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002, align="left")],
                                                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                                          plot_bgcolor='#f9f9f9'))
            st.plotly_chart(fig_network, use_container_width=True)

            st.markdown("---")
            st.subheader("Network Insights (Most Central Topics):")
            if topic_network_graph.number_of_nodes() > 1:
                sorted_degree_centrality = sorted(centrality.items(), key=lambda item: item[1], reverse=True)
                top_n_centrality = min(5, len(sorted_degree_centrality))
                if top_n_centrality > 0:
                    centrality_cols = st.columns(top_n_centrality)
                    for i, (node_id_cent, centrality_val_cent) in enumerate(sorted_degree_centrality[:top_n_centrality]):
                        label_cent = topic_labels_config.get(int(node_id_cent), f"Topic {node_id_cent}")
                        centrality_cols[i].metric(label=f"{label_cent}", value=f"{centrality_val_cent:.3f}")
                else: st.info("Not enough central topics to display metrics.")
        else:
            st.warning("The loaded topic network graph has no nodes or edges to display after processing. Check co-occurrence threshold or graph generation process in your notebook.")
    else:
        st.warning("Topic network graph artifact not loaded or not available. Cannot display this section.")