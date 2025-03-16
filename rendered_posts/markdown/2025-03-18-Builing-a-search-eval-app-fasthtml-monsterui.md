Isaac Flath

AboutBlogRSS

Subscribe

AboutBlogRSS

Subscribe

# Building a Search Evaluation App with FastHTML and MonsterUI

Create a professional web application to evaluate search algorithms without writing CSS or JavaScript

## Contents

The SetupReal-World Problem or ChallengeWhat This Post Will DeliverBackground and FoundationLet's buildSetupDatabaseApplicationSearching and Saving FeedbackViewing Evaluation ResultsPutting It All Together

## The Setup

### Real-World Problem or Challenge

When building search functionality for applications, developers often struggle to systematically evaluate search quality. You might have implemented vector search, BM25, or hybrid approaches, but how do you know which one works best for your specific use case? Creating evaluation interfaces typically requires significant frontend expertise, leaving many Python developers stuck with makeshift command-line tools or spreadsheets to track search performance.

### What This Post Will Deliver

By the end of this tutorial, you'll be able to build a complete search evaluation web application using Python, FastHTML, and MonsterUI. You'll learn how to create a professional-looking interface where you can query different search algorithms, view results side-by-side, and systematically rate and annotate them—all without writing a single line of CSS or JavaScript. Here's a quick preview of what we'll build:

### Background and Foundation

This tutorial assumes you're comfortable with Python but may not have much experience with web development.

Before we dive in, you should understand:

  * Basic Python concepts (functions, classes, list comprehensions)
  * Simple database operations and sql queries (we'll use SQLite through a simplified API)
  * Some familiarity with how web applications work

## Let's build

### Setup

Let's start with importing the key libraries we will need. FastHTML is a Python library that simplifies web app creation by combining HTMX, Starlette, and HTTP fundamentals. MonsterUI extends this with beautiful, ready-to-use UI components that eliminate the need for CSS expertise.

[code]

    from datetime import datetime
    from pathlib import Path
    from fasthtml.common import *
    from monsterui.all import *
[/code]

We will be using a search algorithms that were defined in my Retreival 101 blog post. If you are interested in the details of how those are implemented please check out that blog post.

[code]

    # Import search algorithms - each uses a different retrieval strategy
    from retrieval_scripts.search_blog import (vector_search, bm25_search, hybrid_search, search_blog_posts)
[/code]

### Database

We will be using a library called fastlite for our database management, which allows for simple concise database management in python. To define a SQL table schema we can use python classes.

[code]

    class Query:
        id: int          # Primary key
        query_text: str  # The actual query text
        created_at: str  # Timestamp when the query was created
    
    class Evaluation:
        id: int              # Primary key
        query_id: int        # Foreign key to Query
        result_id: str       # ID of the evaluated search result
        relevance_score: int # Relevance rating (1-5)
        notes: str           # Evaluator's notes
        created_at: str      # Timestamp when the evaluation was created
[/code]

With our schemas we can connect to our databased and the tables.

[code]

    # fastlite uses SQLite under the hood but with a simpler API
    db_path = Path("search_evaluations.db")
    db = database(db_path)
    # Create tables based on class definitions - no SQL or ORM setup needed
    db.queries     = db.create(Query,      pk='id')
    db.evaluations = db.create(Evaluation, pk='id', foreign_keys=('query_id',))
[/code]

> 💡 Tip: `db.create` in `fastlite` is creating a _connection_ to a table. It either connects to the existing table if it exists, or creates a table with the schema provided if not.

Query  
---  
id| int| Primary Key  
query_text| str| Search query  
created_at| str| Timestamp  
  
←FK→

Evaluation  
---  
id| int| Primary Key  
query_id| int| FK → Query.id  
result_id| str| Result ID  
relevance_score| int| Rating (1-5)  
notes| str| Notes  
created_at| str| Timestamp  
  
We need a variety of queries to save and store in sqlite. Let's write a few of them so you can understand how interactions with the database work in fastlite.

`fastlite` handles the insertion and returns the object back to you. You can use the classes defined above to construct the data to insert.

[code]

    def save_query(query_text):
        """Save a query to the database and return its ID."""
        now = datetime.now().isoformat()
        query = db.queries.insert(Query(query_text=query_text, created_at=now))
        return query.id
    
    # rt makes it a route so it can be access and called in your app too
    @rt
    def save_evaluation(query_id:str, result_id:str, rating:int=0, notes:str=""):
        """Save an evaluation to the database."""
        now = datetime.now().isoformat()
        evaluation = db.evaluations.insert(Evaluation(
            query_id=query_id, result_id=result_id, 
            relevance_score=rating, notes=notes, created_at=now))
        return evaluation.id
[/code]

[code]

    save_query("My Query")
[/code]

1

[code]

    db.queries()
[/code]

[Query(id=1, query_text='My Query', created_at='2025-03-14T23:34:36.890074')]

[code]

    save_evaluation(1, 0, 5, "Very cool!")
[/code]

1

[code]

    db.evaluations()
[/code]

[Evaluation(id=1, query_id=1, result_id='0', relevance_score=5, notes='Very cool!', created_at='2025-03-14T23:37:26.332980')]

`fastlite` lets you easily use WHERE clauses from a table connection object for filtering.

[code]

    def get_evaluations_for_query(query_id):
        """Get evaluations for a specific query."""
        evaluations = db.evaluations('query_id=?',[query_id])
        # Sort by relevance to show best results first - key function handles None values
        return sorted(evaluations, key=lambda x: x.relevance_score or 0, reverse=True)
[/code]

[code]

    get_evaluations_for_query(1)
[/code]

[Evaluation(id=1, query_id=1, result_id='0', relevance_score=5, notes='Very cool!', created_at='2025-03-14T23:37:26.332980')]

For more complicated queries, `fastlite` allows raw SQL.

[code]

    def get_evaluation_stats():
        """Get statistics for queries with evaluations."""
        return db.q(f"""
            SELECT q.id, q.query_text, 
                   COUNT(e.id) as eval_count, 
                   AVG(e.relevance_score) as avg_score
            FROM query q
            LEFT JOIN evaluation e ON q.id = e.query_id
            GROUP BY q.id
            HAVING eval_count > 0
            ORDER BY q.created_at DESC
        """)
    get_evaluation_stats()
[/code]

[{'id': 1, 'query_text': 'My Query', 'eval_count': 1, 'avg_score': 5.0}]

[code]

    def get_all_evaluations():
        """Get all evaluations with query text."""
        return db.q(f"""
            SELECT q.query_text, e.result_id, e.relevance_score, e.notes, e.created_at
            FROM evaluation e
            JOIN query q ON e.query_id = q.id
            ORDER BY q.id, e.relevance_score DESC
        """)
    get_all_evaluations()
[/code]

[{'query_text': 'My Query', 'result_id': '0', 'relevance_score': 5, 'notes': 'Very cool!', 'created_at': '2025-03-14T23:37:26.332980'}]

### Application

FastHTML creates an app that can be served. So let's create one!

  * `app`: The main application object that has everything needed to actually run and deploy the app
  * `rt`: A decorator registers routes to `app`

> 💡 Tip: `Theme.blue.headers()` adds the needed headers for MonsterUI which lets you simplify the app a lot by giving strong default styles and convenient helper functions for layout, style, and more.
[code]

    app, rt = fast_app(hdrs=Theme.blue.headers(), live=True)
[/code]

How do we create UI in python? Well, with FastHTML!

If you've done web dev in python before you may have had bad experiences with frameworks that create huge abstractions you can't modify. This is NOT what fasthtml does. The FastHTML library is a 1:1 mapping. `A("Search") == <a>Search</a>` and `Div() == <div></div>`. There's no mysteries

[code]

    Div(A("Search", href='/my-route'))
[/code]

[code]

    <div>
    <a href="/my-route">Search</a></div>
    
    
[/code]

MonsterUI adds python functions on top of that for convenience, but is still very close to the foundations! For example a `Subtitle` is a very simple convenience abstration for HTML.

[code]

    Subtitle()
[/code]

[code]

    <p class="text-gray-500 dark:text-gray-200 text-sm mt-1.5"></p>
    
    
[/code]

More complicated functions create more complicated components for you, but it's still quite close to the foundations of the web. `NavBar` for example using tailwind (common css library) to create a navbar that collapses to a hamburger menu on mobile for you. It's still quite close to the foundations so you can code it yourself, but it's convenient not to (because NavBars are hard to get right as you can see!)

[code]

    NavBar()
[/code]

[code]

    <div>
      <div class="monster-navbar p-4 navbar-underline">
        <div class="flex justify-between items-center w-full">
          <h3 class="uk-h3 ">Title</h3>
    <a class="md:hidden" data-uk-toggle="target: #__VerV2LJSIyA3FhuzeOiIg; cls: hidden"><uk-icon icon="menu" height="30" width="30"></uk-icon></a>      <div class="items-center space-x-4 hidden md:flex"></div>
        </div>
      </div>
      <div class="flex flex-col items-center justify-center space-y-4 p-4 navbar-underline hidden md:hidden monster-navbar" id="__VerV2LJSIyA3FhuzeOiIg"></div>
    </div>
    
    
[/code]

So let's create a navigation bar for our site that have a title and a couple pages.

[code]

    def EvalNavBar():
        """Create navigation bar with links to main app sections."""
        return NavBar(
            # FastHTML components map 1:1 with HTML tags
            # A("Search", href="/") -> <a href="/">Search</a>
            A("Search", href="/",              cls=AT.primary),
            A("Evals",  href=view_evaluations, cls=AT.primary),
            # MonsterUI components can be nested to create complex UI elements
            brand=Div(
                H3("Search Evaluation Tool"), 
                Subtitle("Query, annotate, and rank search results")))
[/code]

[code]

    def layout(content):
        """Wrap content in consistent page layout with navigation."""
        return Div(EvalNavBar(), Container(content))
[/code]

### Searching and Saving Feedback

Lets start by defining what we want our main page to be. Put simply, we want a form for users to enter their query into, and a place for the results of that query to go. We can make our page with the `@rt` decorator.

[code]

    @rt
    def index():
        """Render the main search page."""
        return layout(Div(search_form(), Div(id="search-results")))
[/code]

Let's define this `search_form` we need!

We will need a drop down to select different options. We can abstract out the options creation into a function to keep our code clean and avoid unneccesary repetition.

[code]

    def _Option(label, selected=False):
        return Option(label, value=label.lower(), selected=selected)
[/code]

Now we need to create a form for the user to type in their query, what method they are testing, and how many results they want returned to them.

[code]

    def search_form(query=""):
        """Create search form with query input and search options."""
        return Card(
            Form(
                # MonsterUI's Label* components combine labels with inputs for cleaner UI
                LabelInput("Search Query", id="query", placeholder="Enter your search query...", value=query),
    
                LabelSelect(
                    # An Option component is used to create a dropdown option
                    _Option("Rerank", selected=True),
                    # We can use common python patterns to create a list of options
                    *map(_Option, ("Vector", "BM25", "Hybrid Search")),
                    label="Search Method", id="search_method"),
    
                LabelInput("Number of Results", type="number", id="top_k",min="1", max="20", value="5"),
    
                # ButtonT.primary is an enum for styling - w-full makes button full width
                # This is combining MonsterUI styles with tailwind classes giving lots of control over the style
                Button("Search", cls=(ButtonT.primary,'w-full')),
    
                # HTMX attributes enable dynamic updates without full page reloads
                # Make a POST request to the search route and update the #search-results div with the response
                hx_post=search, hx_target="#search-results"))
[/code]

A fasthtml route is a regular python function with a `@rt` decorator. FastHTML will automatically parse items in the request to arguments that match the name and type hint to make them accessible in python. So lets create the `search` route that the form above is making a POST request to.

Note: This search function returns a `search_result_component` for each result. We will define that later.

[code]

    @rt
    def search(query: str = "", search_method: str = "vector", top_k: int = 5):
        """Execute search and return formatted results."""
        if not query: return Card(P("Please enter a search query", cls=TextT.error))
        
        # Avoid duplicate queries in the database by checking if it exists first
        existing_queries = db.queries('query_text=?',[query])
        query_id = existing_queries[0].id if existing_queries else save_query(query)
        
        # Use the appropriate search function based on the value from the user form
        match search_method:
            case "vector":  results = vector_search(query).sort_values('vector_score', ascending=False).head(top_k)
            case "bm25":    results = bm25_search(query, vector_search(query)).sort_values('bm25_score', ascending=False).head(top_k)
            case "hybrid":  results = hybrid_search(query, top_k=top_k)
            case 'rerank':  results = search_blog_posts(query, top_k=top_k)
            case _: raise ValueError(f"Invalid search method: {search_method}")
                
        # Return the results of the query and associated information
        # Returns a `search_result_component` for each result
        return Div(
            H2(f"Search Results for: '{query}'"),
            P(f"Method: {search_method.capitalize()}"),
            *[search_result_component(results.iloc[i], query_id) for i in range(len(results))])
[/code]

Great! Now we have a `search` route that when a POST request is issued against it will do the appropriate search and return the results in a `search_result_component`, so let's define that!

[code]

    def rating_labels(rating):
        """Convert numeric ratings to descriptive text so users can see what each button means"""
        labels = {1: "Not relevant", 2: "Slightly relevant", 3: "Moderately relevant",
                  4: "Relevant", 5: "Highly relevant"}
        return f"{rating} - {labels[rating]}"
    
    def search_result_component(result, query_id):
        """Create a card for a search result with evaluation controls."""
    
        token_count = result.get('token_count', 0)
        # Visual cues help evaluators quickly assess content length
        if token_count < 1000: token_color = TextT.success     # Green = short, definitely good embedding
        elif token_count < 3000: token_color = TextT.warning   # Yellow = medium length, probably good embedding
        else: token_color = TextT.error                        # Red = too long for embeddding
    
        # Different search methods use different scoring mechanisms    
        score = (result.get('rerank_score')   or 
                 result.get('combined_score') or 
                 result.get('vector_score')   or 
                 result.get('bm25_score')     or 
                 0)
        
        # Return a UI component
        return Card(Article(
            # Make it easy to see the title of the post and the section
            ArticleTitle(result['post_title']), H4(result['chunk_title']),
            
            # DivFullySpaced creates a flexbox with items at opposite ends (Layout helpers)
            DivFullySpaced(
                # Add some values that are helpful for evaluating
                P(f"Score: {score:.4f}"   , cls=TextT.muted + TextT.sm),
                P(f"Tokens: {token_count}", cls=token_color + TextT.sm)),
            Divider(),
            # `render_md` converts markdown text to HTML and styles it nicely for you
            P(render_md(result['content'])),
            Divider(),
            # Create a form for user to put their feedback they want saved in
            Form(
                H4("Relevance Rating:", cls=TextT.medium + TextT.sm),
                Div(
                    # Label* inputs combines a user input with a label for what it is
                    *[LabelRadio(rating_labels(i), name=f"rating", value=str(i)) for i in range(1, 6)],
                    cls="space-y-2"),
                LabelTextArea("Notes:", id=f"notes", placeholder="Add notes about this result...", rows="2", cls='w-full'),
                Button("Save Evaluation", type="button", cls=ButtonT.primary,
                       # HTMX will send the form data, query_id, and result_id to the `save_evaluation` endpoint
                       hx_post=save_evaluation.to(query_id=query_id, result_id=str(result.name)),
                       hx_swap="none"))))
[/code]

Ok, so we've stored all of the data. We created a `search_form` for users to enter their search query into. That calls the `search` route which does the actual search and stores the results in the database. And that passes the neccesary information to `search_result_component` to allow users to see the results and rate them. All that is put together into a single page for users in `index`.

### Viewing Evaluation Results

We're ready to view the results of our evaluation. Lets create a route for that! We can create a single route that if there is no `query_id` shows a summary (`AllEvalsTable`), and if a `query_id` is provided it shows details on that particular query (`SingleEvalTable`).

We'll define the tables later, but let's get the big picture first.

[code]

    @rt
    def view_evaluations(query_id: int = None):
        """Show either all queries with evaluations or details for a specific query."""
        if not query_id:
            # Overview mode - show all queries with evaluation stats
            queries = get_evaluation_stats()        
            return layout(Card(H2("Queries with Evals"),  AllEvalsTable(queries)))
        else:
            # Detail mode - show evaluations for a specific query
            query = db.queries[query_id]
            evaluations = get_evaluations_for_query(query_id)
            return layout(Div(H4(f"Evals for: '{query.query_text}'", cls='mb-6'), Card(SingleEvalTable(evaluations))))
[/code]

So now we need to create the tables!

[code]

    def AllEvalsTable(queries):
        """Create a table showing all queries with evaluation statistics using TableFromDicts."""
        headers = ["Query", "Evaluations", "Avg. Score", "Actions"]
        
        def create_row(query):
            "Pull data from query and format it for viewing"
            return {
                "Query":       query['query_text'],
                "Evaluations": str(query['eval_count']),
                "Avg. Score":  f"{query['avg_score']:.2f}" if query['avg_score'] else "N/A",
                # Actions uses a link (href) to send to the `view_evaluations` route for the specific query
                "Actions":     A("View Details", href=view_evaluations.to(query_id=query['id']), cls=AT.primary)}
        
        rows = list(map(create_row, queries))
        return TableFromDicts(headers, rows)
[/code]

[code]

    def SingleEvalTable(evaluations):
        """Create a table showing evaluations for a single query using TableFromDicts."""
        headers = ["Result ID", "Relevance", "Notes", "Date"]
        
        def create_row(eval):
            "Pull data from an eval and format it for viewing"
            return {
                "Result ID": eval.result_id,
                "Relevance": str(eval.relevance_score),
                "Notes":     eval.notes or "No notes",
                "Date":      eval.created_at}
        
        rows = list(map(create_row, evaluations))
        return TableFromDicts(headers, rows)
[/code]

Then we start the server!

[code]

    serve()
[/code]

## Putting It All Together

Let's review what we've built: a complete search evaluation application with just Python. Our app lets you:

  1. Run queries against different search algorithms
  2. View and compare search results side-by-side
  3. Rate results on a 1-5 relevance scale
  4. Add notes to explain your ratings
  5. View evaluation statistics across queries

The entire application is built with FastHTML and MonsterUI, eliminating the need for CSS or JavaScript knowledge. Here's the complete structure:

[code]

    # Main components:
    # - Database schema and operations (Query and Evaluation classes)
    # - Search functionality (vector_search, bm25_search, hybrid_search)
    # - UI components (EvalNavBar, search_form, search_result_component)
    # - Routes (index, search, view_evaluations)
    # - Evaluation views (AllEvalsTable, SingleEvalTable)
    
    # Start the application
    serve()
    
[/code]

### From Theory to Practice

This search evaluation tool isn't just a demo—it's designed for real-world use. Here are some practical ways to integrate it into your workflow:

**Iterative Algorithm Development** : As you tweak your search algorithms, use this tool to systematically track improvements. The evaluation database provides concrete evidence of which changes actually improve relevance.

**Team Collaboration** : Multiple team members can use the same instance to evaluate results, building a shared understanding of what "good search" means for your specific use case.

**Stakeholder Demos** : When explaining search improvements to non-technical stakeholders, this visual interface is much more compelling than command-line outputs or spreadsheets.

### Key Takeaways and Principles

Building this application demonstrates several powerful principles:

  1. **Separation of concerns** : The search algorithms, database operations, and UI components are cleanly separated, making the code maintainable and extensible.

  2. **Progressive enhancement** : The HTMX-powered interactions provide a modern feel without complex client-side JavaScript.

  3. **Pythonic web development** : FastHTML and MonsterUI let Python developers build web applications without learning new paradigms or languages.

  4. **Data-driven decisions** : Systematic evaluation transforms search development from guesswork into an evidence-based process.

Most importantly, this approach puts the power of building professional web applications into the hands of Python developers who may not have frontend expertise. You can focus on your core strengths—data processing, search algorithms, and Python programming—while still delivering polished user experiences.

### Next Steps for the Reader

Now that you have this foundation, here are some ways to extend and adapt this application:

  * **Add more search algorithms** : Implement additional retrieval methods like dense passage retrieval or learned sparse retrieval.
  * **Expand evaluation metrics** : Add dimensions beyond relevance, such as diversity or freshness of results.
  * **Build visualization dashboards** : Create charts and graphs to visualize evaluation trends over time.
  * **Implement A/B testing** : Modify the app to show results from two algorithms side-by-side for direct comparison.
  * **Add authentication:** Create a system of authentication so you can track what users provide what feedback to follow up with them

I'd love to see your eval application! Share your extensions and adaptations with the community.

If you found this useful, check out my other tutorials on building vector search systems and FastHTML application development and [FastHTML and HTMX](http://localhost:5001/blog/blog_post?fpath=posts%2F2025-04-03-Dynamic UI Interactions with FastHTML and HTMX.ipynb).

Happy searching!

### Stay Updated

Get notified about new posts on AI, web development, and tech insights.

Subscribe

## Contents

The SetupReal-World Problem or ChallengeWhat This Post Will DeliverBackground and FoundationLet's buildSetupDatabaseApplicationSearching and Saving FeedbackViewing Evaluation ResultsPutting It All Together

## Let's Connect

GitHub

LinkedIn

Twitter

© 2025 Isaac Flath • All rights reserved

