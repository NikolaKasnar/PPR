#pragma once
#include <memory>
#include <mutex>
#include <iostream>

class List{
    struct Node{
        mutable std::mutex m;
        int data;
        Node * next;
        Node() : next(nullptr) {}
        Node(int const & value) : data(value), next(nullptr) {}
    };
 
	// Ovdje držimo (dummy) čvor, a ne pokazivač na prvi čvor, radi uniformnosti koda. 
    // Prvi element je head->next!
    Node head;

    public:
       List() : head{} {}
       ~List(); 
       
	   // Ne dozvoljavamo kopiranje
       List(List const &) = delete;
       List & operator=(List const &) = delete;

       // ubaci element na početak liste.
       void push_front(int value);
	   // Ubaci element na kraj liste.
       void push_back(int  value);
	   // Obriši sve elemente s vrijednošću value i vrati broj obrisanih
       int remove(int value); 
       // Svaku nađenu vrijednost old_val zamijeni u new_val.
       // Vrati broj zamjena.
       int find_and_change(int old_val, int new_val);
       // Postoji li element jednak value?
       bool contains(int value) const;
       // Vrati broj elemenata u listi.
       int size() const;
        // Ispiši listu.
       void print(std::ostream & out) const;
};

// VAŠA IMPLEMENTACIJA SVIH METODA.

// Desturktor za listu
List::~List() {
    Node* current = head.next;
    while (current != nullptr) {
        Node* temp = current;
        current = current->next;
        delete temp;
    }
}

// Ovu implementaciju i implementaciju od push_back() sam morao napraviti sa lock_guard
// nadam see da se to ne kosi sa pravilima zadatka ali nazalost nisam stigao drugacije rijesiti
// bez da mi nastane deadlock ili da mi zadnji assert faila
void List::push_front(int value) {
    Node* newNode = new Node(value);
    std::lock_guard<std::mutex> guard(head.m); // Zakljucamo head clan sve dok funckija ne zavrsi

    newNode->next = head.next;
    head.next = newNode;
}

void List::push_back(int value) {
    Node* newNode = new Node(value);
    Node* current = &head;
    std::lock_guard<std::mutex> guard(head.m);

    while (current->next != nullptr) {
        current = current->next;
    }
    current->next = newNode;
}

int List::remove(int value) {
    int count = 0;
    Node* prev = &head;
    prev->m.lock();
    Node* current = prev->next;
    while (current != nullptr) {
        current->m.lock();
        if (current->data == value) {
            prev->next = current->next;
            current->m.unlock();
            delete current;
            count++;
            current = prev->next;
        } else {
            prev->m.unlock();
            prev = current;
            current = current->next;
        }
    }
    prev->m.unlock();
    return count;
}

int List::find_and_change(int old_val, int new_val) {
    int count = 0;
    Node* current = &head;
    while (current->next != nullptr) {
        current->m.lock();
        if (current->data == old_val) {
            current->data = new_val;
            count++;
        }
        Node* temp = current;
        current = current->next;
        temp->m.unlock();
    }
    current->m.lock();
    if (current->data == old_val) {
        current->data = new_val;
        count++;
    }
    current->m.unlock();
    return count;
}

bool List::contains(int value) const {
    Node* current = head.next;
    while (current != nullptr) {
        current->m.lock();
        if (current->data == value) {
            current->m.unlock();
            return true;
        }
        Node* temp = current;
        current = current->next;
        temp->m.unlock();
    }
    return false;
}

int List::size() const {
    int count = 0;
    Node* current = head.next;
    while (current != nullptr) {
        current->m.lock();
        count++;
        Node* temp = current;
        current = current->next;
        temp->m.unlock();
    }
    return count;
}

void List::print(std::ostream & out) const {
    Node* current = head.next;
    while (current != nullptr) {
        current->m.lock();
        out << current->data << " ";
        Node* temp = current;
        current = current->next;
        temp->m.unlock();
    }
}