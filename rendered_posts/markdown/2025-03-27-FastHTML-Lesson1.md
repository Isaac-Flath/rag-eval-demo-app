Isaac Flath

AboutBlogRSS

Subscribe

AboutBlogRSS

Subscribe

# FastHTML Lesson 1

Learn to build web applications with FastHTML using Python

## Contents

Getting Started with FastHTML: Web Apps Made Easy for Python DevelopersWhat is FastHTML?Setting Up Your EnvironmentApplication 1: Temperature ConverterWhat should you be able to answer?Deep DiveApplication 2: Todo List AppQuestions to answer?Diving deeperNext StepsConclusion

# Getting Started with FastHTML: Web Apps Made Easy for Python Developers

If you've been writing Python scripts but never ventured into web development, you're in for a treat. FastHTML is a Python library that makes building web applications surprisingly straightforward, even if you've never touched HTML, CSS, or JavaScript before.

In this tutorial, I'll walk you through creating two simple web applications: a temperature converter and a todo list. By the end, you'll have a solid foundation for building your own web apps with Python.

This is the first in a series of tutorials designed to help you teach yourself FastHTML by learning from examples.

## What is FastHTML?

FastHTML combines several powerful tools (Starlette, Uvicorn, HTMX, and FastTags) into a single library that lets you create web applications using just Python. No need to learn multiple languages or frameworks - if you know Python, you're already most of the way there.

## Setting Up Your Environment

First, let's install FastHTML:

[code]

    # pip install python-fasthtml monsterui
[/code]

MonsterUI is an optional but recommended companion library that provides beautiful styling with minimal effort.

## Application 1: Temperature Converter

Let's start with a simple temperature converter that converts between Celsius and Fahrenheit. Create a new file called `temp_converter.py`:

[code]

    ## File: temp_converter.py
    
    # FastHTML is the web app framework
    from fasthtml.common import *
    # MonsterUI is a UI library focused on developer experience
    from monsterui.all import *
    
    # This is how you initialize a fastHTML app
    app, rt = fast_app(hdrs=Theme.blue.headers())
    
    # @rt tells your app to make the function a route
    # A route is a URL that can be accessed
    @rt
    def index():
        return Titled("Temperature Converter",
                # Create a form with an input and a button
                Form(Input("Celsius", name="celsius", type="number", placeholder="Enter temperature in Celsius"),
                     Button("Convert to Fahrenheit"), 
                     
                     # When button is pressed call the `to_fahrenheit function
                     # Put the result in the id `result`
                     hx_post=to_fahrenheit, hx_target="#result",
                     
                     # hx_trigger and hx_swap are also very important, but the defaults are fine in this case!
                     # hx_trigger='submit', hx_swap='innerHTML'
                    ),
                     # A Div is an container that anything can go in
                     # We give it an id so we can reference it later
                     # After clicking the button it'll get filled in
                     Div(id="result"))
    
    # The to_fahrenheit function needs to be a route so the app can access it!
    @rt
    def to_fahrenheit(celsius: float = 0):
        fahrenheit = (celsius * 9/5) + 32
        return P(f"{celsius}°C = {fahrenheit:.2f}°F")
    
    # This actually launches the app!
    serve()
[/code]

Great, now run it with `python temp_converter.py` and open `localhost:5001` in your browser. You've got a running app!

Let's break down what it's doing and how it works.

### What should you be able to answer?

  1. How is a fasthtml app initialize and created?
  2. How do you tell your script to launch the app?
  3. What is a route?
  4. How do you turn a regular python function into one?
  5. What does hx_post do?
  6. What does hx_target do?

### Deep Dive

  * **HTML Tags** : Input, Button, Div, and P are all HTML elements. FastHTML lets you create these using Python functions with the same names. If you're curious about what HTML tags exist and how they work, the MDN Web Docs is an excellent reference.

  * **HTMX and the main 4 attributes** : HTMX is the magic that makes our app interactive without writing JavaScript. The four main attributes you'll use most often are:

    * `hx_post` or `hx_get`: Specifies what URL to call when triggered
    * `hx_target`: Specifies which element to update with the response
    * `hx_trigger`: Defines what event triggers the request (default is "submit" for forms, and "click" for most other things)
    * `hx_swap`: Controls how content is swapped in (default is "innerHTML") 
      * `innerHTML` swaps what is inside of an HTML element
      * `outerHTML` swaps the an entire HTML element
      * `beforeend` puts things before the end of an element
      * There's many more options!

Learn more at the HTMX documentation.

  * **CSS Selectors** : When using `hx_target` or targeting elements, you'll use CSS selectors. The most common are:

    * `#id` \- Selects an element with a specific ID (like `#result`)
    * `.class` \- Selects elements with a specific class
    * `tag` \- Selects all elements of a specific tag (like `body`)

MonsterUI provides many pre-styled components and themes so you don't need to write CSS yourself.

  * **MonsterUI** : In our example, we used MonsterUI that does a lot of default styling for you. I recommend looking through every page for examples in the API reference and examples in the monsterui docs to get an idea of what's possible.

  * **Form Handling** : FastHTML automatically converts form data to Python types. Notice how our `to_fahrenheit` function accepts a `celsius: float` parameter that's automatically extracted from the form.

Now let's build something a bit more complex to see how these concepts come together in a more practical application.

## Application 2: Todo List App

Now let's build something a bit more complex - a todo list app that lets you add and delete tasks.

Create a new file called `todo_app.py`:

[code]

    ## File: todo_app.py
    
    # Import everything we need
    from fasthtml.common import *
    from monsterui.all import *
    
    # Initialize our app with the blue theme
    app, rt = fast_app(hdrs=Theme.blue.headers())
    
    # We'll store our todos in a simple list for now
    todos_list = []
    todo_id_counter = 0
    
    # Our main page route
    @rt
    def index():
        return Titled("Todo List App",
            # Create a form to add new todos
            Form(
                # Input for the todo title
                Input(name="title", placeholder="Enter a new task", required=True),
                # Button to submit
                Button("Add Todo"),
                # When the form is submitted, call the add_todo function
                # and put the result at the end of the todo-list element
                hx_post=add_todo, 
                hx_target="#todo-list", 
                hx_swap="beforeend"),
            # This div will contain our list of todos
            Div(
                # Create a heading for our list
                H3("My Tasks:"),
                # Create an unordered list with an id so we can target it
                Ul(id="todo-list")))
    
    # Function to add a new todo
    @rt
    def add_todo(title: str):
        global todo_id_counter
        # Increment our ID counter
        todo_id_counter += 1
        # Add the todo to our list
        todos_list.append({"id": todo_id_counter, "title": title})
        
        # Return a list item with the todo and a delete button
        return Li(
            # Display the todo title
            Span(title),
            # Add a delete button
            Button("Delete", 
                # When clicked, call delete_todo with this todo's ID
                hx_post=delete_todo.to(id=todo_id_counter), 
                # Target the list item for removal
                hx_target=f"#todo-{todo_id_counter}",
                # Remove the entire list item when deleted
                hx_swap="outerHTML",
                # Add some styling with MonsterUI
                cls=ButtonT.destructive
            ),
            # Give the list item an ID so we can target it for deletion
            id=f"todo-{todo_id_counter}"
        )
    
    # Function to delete a todo
    @rt
    def delete_todo(id: int):
        global todos_list
        # Remove the todo from our list
        todos_list = [todo for todo in todos_list if todo["id"] != id]
        # We don't need to return anything since we're removing the element
    
    # Launch the app
    serve()
[/code]

### Questions to answer?

  1. How does the app store state?
  2. How do you create a form that adds new items to a list?
  3. How do you target specific elements for updates or removal?
  4. What's the difference between different `hx_swap` values like "innerHTML", "outerHTML", and "beforeend"?
  5. How do you pass parameters to route functions when calling them with HTMX?
  6. How do you add styling to a button with MonsterUI?

### Diving deeper

  * **State Management** : In our simple app, we're using global variables (`todos_list` and `todo_id_counter`) to maintain state. For real applications, you'd want to use a database or more robust state management solution. We'll explore fastlite and how to do this in a future post.

  * **Dynamic IDs** : Notice how we create unique IDs for each todo item (`id=f"todo-{todo_id_counter}"`) and then target them specifically for deletion. This pattern is essential for manipulating individual elements in a list.

  * **Route Parameters** : The `.to()` method lets you pass parameters to route functions. In `hx_post=delete_todo.to(id=todo_id_counter)`, we're pre-configuring the route call with the specific todo ID.

  * **UI Components** : MonsterUI provides styled components like `ButtonT.destructive` that give visual cues about actions (red for delete buttons). This improves user experience without learning CSS.

  * **List Manipulation** : The pattern of adding items with `hx_swap="beforeend"` and removing them with `hx_swap="outerHTML"` is common for list operations. "beforeend" adds content at the end of the target element, while "outerHTML" replaces the entire element (including itself).

  * **Form Handling** : FastHTML automatically extracts form data and converts it to the appropriate Python types based on the function's type hints (`title: str`).

## Next Steps

Now that you've built two simple apps, here are some ideas to expand your FastHTML skills:

  1. **Add persistence** : Use FastHTML's built-in database support to save your todos between server restarts
  2. **Add user authentication** : Create a login system for your app
  3. **Explore more MonsterUI components** : Try using different components like `Alert`, `Modal`, or `Tabs`
  4. **Create a multi-page app** : Add more routes and link between them

## Conclusion

FastHTML makes web development accessible to Python developers without requiring knowledge of HTML, CSS, or JavaScript. By combining Python's simplicity with the power of modern web technologies, you can build interactive web applications with minimal code.

The examples we've built are just the beginning. As you get more comfortable with FastHTML, you'll find it's capable of building sophisticated web applications while keeping your codebase clean and maintainable.

Stay tuned - this is the first in a series of posts designed to help you teach yourself FastHTML.

Happy coding!

### Stay Updated

Get notified about new posts on AI, web development, and tech insights.

Subscribe

## Contents

Getting Started with FastHTML: Web Apps Made Easy for Python DevelopersWhat is FastHTML?Setting Up Your EnvironmentApplication 1: Temperature ConverterWhat should you be able to answer?Deep DiveApplication 2: Todo List AppQuestions to answer?Diving deeperNext StepsConclusion

## Let's Connect

GitHub

LinkedIn

Twitter

© 2025 Isaac Flath • All rights reserved

