class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def add_node(self, data):
        """Add a node to the end of the list."""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return

        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def print_list(self):
        """Print all elements in the list."""
        if not self.head:
            print("List is empty.")
            return

        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

    def delete_nth_node(self, n):
        """Delete the nth node (1-based index) from the list."""
        if not self.head:
            raise IndexError("Cannot delete from an empty list.")

        if n <= 0:
            raise IndexError("Index must be a positive integer.")

        if n == 1:
            print(f"Deleting node at position {n} with value {self.head.data}")
            self.head = self.head.next
            return

        current = self.head
        prev = None
        count = 1

        while current and count < n:
            prev = current
            current = current.next
            count += 1

        if not current:
            raise IndexError("Index out of range.")

        print(f"Deleting node at position {n} with value {current.data}")
        prev.next = current.next


# Test the LinkedList implementation
if __name__ == "__main__":
    ll = LinkedList()

    # Add sample data
    for value in [10, 20, 30, 40, 50]:
        ll.add_node(value)

    print("Original list:")
    ll.print_list()

    try:
        ll.delete_nth_node(3)  # Delete the 3rd node (value = 30)
    except IndexError as e:
        print(e)

    print("List after deleting 3rd node:")
    ll.print_list()

    try:
        ll.delete_nth_node(10)  # Out of range
    except IndexError as e:
        print(f"Error: {e}")

    try:
        empty_ll = LinkedList()
        empty_ll.delete_nth_node(1)  # Deleting from an empty list
    except IndexError as e:
        print(f"Error: {e}")
