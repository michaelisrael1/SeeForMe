Milestone 1: Initial Local Implementation
Initial local implementation using YOLOv5 and CV2 was completed. This provided a good Proof of Concept (PoC) for the direction we wanted the project to take.

Milestone 2: API and Initial Cloud Architecture
A local YOLOv5 API was implemented on a desktop computer. This allowed the Raspberry Pi to process video faster and reduce the consumption of AWS tokens.
However, the YOLOv5 identification was lackluster and didn't provide in-depth analysis. We architected the AWS implementation but did not connect it with the UI yet.

Milestone 3: UI Pivot and Object Overlay
We focused on the overlay that identifies objects within the screen. We pivoted to using a UI again, as we were initially going to have it UI-less.
We further attempted to implement the system on the Raspberry Pi but experienced issues with audio.

Milestone 4: UI Completion and AWS Integration
The UI development was completed based on the paper prototypes.
We removed local functionality, making AWS the only processing option. Functionality was added for users to input their credentials into the UI.
